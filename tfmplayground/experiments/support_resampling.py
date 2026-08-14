"""Support-resampling variance of the nanoTabPFN encoder, and its link to query risk.

Two prior trials tried to make nanoTabPFN report epistemic uncertainty through a
*learned* head and both stalled: the slot posterior's mutual information was
anti-correlated with mistakes (see ``MEAN_PRESERVING_BAYESIAN_TRIAL.md``), and the
continuous posterior's dispersion gate stayed flat to within a few percent across
conditions whose true epistemic content differs by 0.17 nats, at every capacity
budget tried (see ``CONTINUOUS_BAYESIAN_UNCERTAINTY_TRIAL.md``). Both results are
ambiguous between "the frozen encoder carries no epistemic information" and "the
learned head cannot extract it".

This module tests the representation directly, without training anything. The
support set is resampled (bootstrap, stratified subsample, or a resampling
approximation to the Bayesian bootstrap) into ``B`` members, one batched forward
pass through the shared backbone gives every member's prediction and every
layer's query representation, and dispersion across members is read off as a
frequentist stability statistic -- not the model's self-reported belief. Two
additional arms need no resampling at all: they read the model's own posterior by
conditioning on a hypothetical label for one query and checking how much that
moves another query's (or the same query's) belief.

None of these estimators sees any query label. Query labels enter only through
the caller-supplied scoring targets, kept in a disjoint code path.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tfmplayground.models.continuous_posterior import binary_entropy, stratified_subset_indices
from tfmplayground.models.nanotabpfn import NanoTabPFNModel

#: Resampling schemes compared in the sweep.
SCHEMES = ("bootstrap", "subsample", "bayesian_bootstrap")
#: Numerical floor so ratios and logs stay finite.
EPSILON = 1e-8


def split_fit_and_heldout(
    support_x: torch.Tensor, support_y: torch.Tensor, heldout_size: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split one episode's support rows into a fitting pool and a held-out pool.

    The held-out rows never enter any ensemble member; they exist only to score
    members on genuinely unseen data. Returns ``(fit_x, fit_y, heldout_x,
    heldout_y)``.
    """
    total = support_x.shape[0]
    if not 0 < heldout_size < total:
        raise ValueError(f"heldout_size must lie in (0, {total}), found {heldout_size}.")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    permutation = torch.randperm(total, generator=generator).to(support_x.device)
    heldout_indices, fit_indices = permutation[:heldout_size], permutation[heldout_size:]
    return (
        support_x[fit_indices],
        support_y[fit_indices],
        support_x[heldout_indices],
        support_y[heldout_indices],
    )


# --------------------------------------------------------------------------- #
# Member index sets
# --------------------------------------------------------------------------- #
def _member_indices(
    scheme: str,
    fit_size: int,
    members: int,
    generator: torch.Generator,
    *,
    fraction: float = 0.8,
) -> torch.Tensor:
    """Return ``(members, draw_size)`` row indices into a fitting pool of size ``fit_size``.

    ``bootstrap`` draws ``fit_size`` rows with replacement per member, so about
    63% of rows are unique -- the literal question, but confounded with a smaller
    effective context. ``subsample`` draws a fixed ``fraction`` without
    replacement, stratified by label so class balance is preserved; the draw
    size is identical across members, which is what makes the layer captures
    stackable in one batched forward pass. ``bayesian_bootstrap`` realizes a
    ``Dirichlet(1)`` weight vector and resamples ``fit_size`` rows proportional
    to it -- a resampling approximation, documented as such, because
    ``NanoTabPFNModel`` accepts no per-row weights.
    """
    if scheme == "bootstrap":
        return torch.randint(0, fit_size, (members, fit_size), generator=generator)
    if scheme == "subsample":
        # Stratification needs support labels, so this scheme is dispatched from
        # the caller (which has them); see ``_stratified_member_indices``.
        raise ValueError("subsample indices require label-aware dispatch.")
    if scheme == "bayesian_bootstrap":
        weights = torch.distributions.Dirichlet(torch.ones(fit_size)).sample((members,))
        return torch.multinomial(weights, fit_size, replacement=True, generator=generator)
    raise ValueError(f"Unknown scheme {scheme!r}; expected one of {SCHEMES}.")


def _stratified_member_indices(
    support_y: torch.Tensor, members: int, fraction: float, seed: int
) -> torch.Tensor:
    """``(members, draw_size)`` stratified subsample indices, one draw per member.

    The draw size depends only on ``fraction`` and the label counts, so it is
    identical across members; only which rows are chosen differs.
    """
    draws = []
    for member in range(members):
        generator = torch.Generator(device="cpu").manual_seed(seed * 100_003 + member)
        draws.append(stratified_subset_indices(support_y, fraction, generator))
    sizes = {draw.numel() for draw in draws}
    if len(sizes) != 1:
        raise RuntimeError("Stratified draws disagreed on size; this should be impossible.")
    return torch.stack(draws)


# --------------------------------------------------------------------------- #
# Layer capture
# --------------------------------------------------------------------------- #
class LayerCapture:
    """Read-only forward hooks that record each transformer block's query embeddings.

    Hooks read ``output[:, split:, -1, :]`` -- the target-column embedding of the
    query rows, exactly what the existing uncertainty heads consume -- and never
    modify the output, so ``NanoTabPFNModel.encode_table`` is unaffected by
    attaching or detaching them. Captures are detached by default; pass
    ``keep_graph=True`` to keep them attached to the autograd graph, which is
    what the decoder-projection gradient needs -- in that mode the *whole*
    block output is captured, unsliced, because the next block consumes the
    whole tensor and not a sliced copy of it: a slice taken inside the hook
    would be a dead-end leaf with no path back to the decoder's output, and
    ``torch.autograd.grad`` would correctly refuse to differentiate through it.
    Callers slice ``[:, split:, -1, :]`` themselves after the gradient call.
    ``keep_graph`` intentionally does *not* call ``.retain_grad()`` -- the
    gradient is read with ``torch.autograd.grad(outputs, inputs=captures)``
    instead of ``.backward()``, so it never touches ``.grad`` on the model's own
    parameters.
    """

    def __init__(self, model: NanoTabPFNModel, split: int, *, keep_graph: bool = False):
        self.split = split
        self.keep_graph = keep_graph
        self.captures: list[torch.Tensor] = []
        self._handles = [block.register_forward_hook(self._hook) for block in model.transformer_blocks]

    def _hook(self, _module, _inputs, output: torch.Tensor) -> None:
        if self.keep_graph:
            self.captures.append(output)
        else:
            self.captures.append(output[:, self.split :, -1, :].detach().clone())

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()

    def __enter__(self) -> LayerCapture:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.remove()


# --------------------------------------------------------------------------- #
# Resample ensemble
# --------------------------------------------------------------------------- #
@dataclass
class ResampleEnsemble:
    """One resampling scheme's members for one episode, plus the derived statistics.

    Attributes:
        scheme: which resampling scheme produced the members.
        member_probabilities: ``(members, query)`` class-1 probability per member.
        layer_query_embeddings: one ``(members, query, embedding)`` tensor per
            transformer block, in order.
        base_positive: ``(query,)`` full-context vanilla class-1 probability,
            used only as a reference point; never involved in dispersion itself.
        base_layer_embeddings: one ``(query, embedding)`` tensor per block, from
            the same full-context forward pass, used for the decoder-projection
            gradient.
        base_layer_gradients: one ``(query, embedding)`` tensor per block: the
            gradient of the summed positive-class probability with respect to
            that layer's base-point query embedding. First-order (the decoder is
            a two-layer MLP with a GELU nonlinearity, not linear), evaluated once
            per episode. ``None`` until :func:`decoder_gradient` is called.
        heldout_member_probabilities: ``(members, heldout)`` class-1 probability
            per member on the held-out *support* rows passed to
            :func:`build_ensemble`, or ``None`` if none were given. These rows
            were never in any member's fitting draw, so
            ``Var_b(member_heldout_log_loss)`` is an unbiased per-member
            calibration signal to correlate against query dispersion.
    """

    scheme: str
    member_probabilities: torch.Tensor
    layer_query_embeddings: list[torch.Tensor]
    base_positive: torch.Tensor
    base_layer_embeddings: list[torch.Tensor]
    base_layer_gradients: list[torch.Tensor] | None = None
    heldout_member_probabilities: torch.Tensor | None = None

    @property
    def members(self) -> int:
        return self.member_probabilities.shape[0]

    @property
    def num_layers(self) -> int:
        return len(self.layer_query_embeddings)

    def heldout_member_log_loss(self, heldout_y: torch.Tensor) -> torch.Tensor:
        """``(members,)`` mean binary log loss of each member on the held-out rows."""
        if self.heldout_member_probabilities is None:
            raise RuntimeError("No held-out rows were passed to build_ensemble().")
        positive = self.heldout_member_probabilities.clamp(EPSILON, 1.0 - EPSILON)
        label = heldout_y.to(positive.dtype)[None, :]
        return -(label * positive.log() + (1.0 - label) * (1.0 - positive).log()).mean(dim=-1)

    def probability_dispersion(self) -> torch.Tensor:
        """``(query,)`` variance of the member class-1 probabilities."""
        return self.member_probabilities.var(dim=0, unbiased=False)

    def probability_mutual_information(self) -> torch.Tensor:
        """``(query,)`` ``H(mean_b p_b) - mean_b H(p_b)``, comparable to teacher MI."""
        mean_probability = self.member_probabilities.mean(dim=0)
        return (
            binary_entropy(mean_probability) - binary_entropy(self.member_probabilities).mean(dim=0)
        ).clamp_min(0.0)

    def _deviation(self, layer: int) -> torch.Tensor:
        embeddings = self.layer_query_embeddings[layer]
        return embeddings - embeddings.mean(dim=0, keepdim=True)

    def representation_dispersion(self) -> torch.Tensor:
        """``(layer, query)`` mean squared deviation per embedding dimension."""
        rows = []
        for layer in range(self.num_layers):
            deviation = self._deviation(layer)
            embedding_size = deviation.shape[-1]
            rows.append(deviation.square().sum(dim=-1).mean(dim=0) / embedding_size)
        return torch.stack(rows)

    def scale_free_representation_dispersion(self) -> torch.Tensor:
        """``(layer, query)`` raw dispersion divided by the mean embedding norm.

        LayerNorm makes raw embedding norms layer-dependent, so the un-scaled
        dispersion is not comparable across layers; this version is.
        """
        rows = []
        for layer in range(self.num_layers):
            deviation = self._deviation(layer)
            raw = deviation.square().sum(dim=-1).mean(dim=0)
            scale = self.layer_query_embeddings[layer].square().sum(dim=-1).mean(dim=0).clamp_min(EPSILON)
            rows.append(raw / scale)
        return torch.stack(rows)

    def effective_rank(self) -> torch.Tensor:
        """``(layer, query)`` participation ratio of the member deviation matrix.

        ``1`` when the resampling variance lives entirely in one direction (the
        deviations look like "which hypothesis is active"); ``min(members,
        embedding)`` when it is isotropic (resampling jitter with no structure).
        """
        rows = []
        for layer in range(self.num_layers):
            deviation = self._deviation(layer).transpose(0, 1)  # (query, members, embedding)
            singular_values = torch.linalg.svdvals(deviation)
            squared = singular_values.square()
            participation = squared.sum(dim=-1).square() / squared.square().sum(dim=-1).clamp_min(EPSILON)
            rows.append(participation)
        return torch.stack(rows)

    def decoder_gradient(
        self, decoder: torch.nn.Module, raw_base_captures: list[torch.Tensor], split: int
    ) -> None:
        """Populate ``base_layer_gradients`` from one ``autograd.grad`` call at the base point.

        ``raw_base_captures`` are the *unsliced* per-block outputs from a
        ``keep_graph=True`` :class:`LayerCapture` pass -- gradients are taken
        against those, not against ``base_layer_embeddings``, because the
        pre-sliced query embeddings are dead-end leaves that the next block
        never reads (see :class:`LayerCapture`'s docstring). Query rows never
        attend to each other (``datapoint_attention_stage`` routes test rows
        only to support rows as keys/values), so summing the positive-class
        probability over all queries before differentiating gives exactly
        ``d(probability_q)/d(embedding_q)`` for every query at once, with no
        cross-query contamination -- one gradient call rather than one per
        query. Uses ``torch.autograd.grad``, not ``.backward()``, so the
        model's own parameters never accumulate ``.grad``.
        """
        logits = decoder(raw_base_captures[-1][:, split:, -1, :])[..., :2]
        positive = logits.squeeze(0).softmax(dim=-1)[..., 1]
        gradients = torch.autograd.grad(positive.sum(), raw_base_captures)
        self.base_layer_gradients = [
            gradient[:, split:, -1, :][0].detach().clone() for gradient in gradients
        ]

    def projected_dispersion(self) -> torch.Tensor:
        """``(layer, query)`` resampling variance along the decoder's sensitivity direction.

        Splits total dispersion into the component the decoder is sensitive to
        and its orthogonal complement: comparing this against
        :meth:`representation_dispersion` separates "the encoder represents
        uncertainty but the read-out discards it" from "the encoder never had
        it". The projection direction is unit-normalized, so ``Var(direction .
        deviation) <= trace(Cov(deviation))``, i.e. the ratio to the raw total
        dispersion lies in ``[0, 1]``.
        """
        if self.base_layer_gradients is None:
            raise RuntimeError("Call decoder_gradient() before projected_dispersion().")
        rows = []
        for layer in range(self.num_layers):
            deviation = self._deviation(layer)
            gradient = self.base_layer_gradients[layer]
            direction = gradient / gradient.norm(dim=-1, keepdim=True).clamp_min(EPSILON)
            projected = (deviation * direction[None]).sum(dim=-1)
            rows.append(projected.square().mean(dim=0))
        return torch.stack(rows)

    def projected_ratio(self) -> torch.Tensor:
        """``(layer, query)`` projected dispersion divided by raw total dispersion."""
        total = self.representation_dispersion() * self.layer_query_embeddings[0].shape[-1]
        return self.projected_dispersion() / total.clamp_min(EPSILON)


# --------------------------------------------------------------------------- #
# Building an ensemble
# --------------------------------------------------------------------------- #
def _encode_and_capture(
    model: NanoTabPFNModel, x: torch.Tensor, y: torch.Tensor, split: int, *, num_mem_chunks: int, keep_graph: bool
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Run ``encode_table`` with layer capture; return the query embedding and the captures.

    When ``keep_graph`` is ``False`` the captures are already sliced to the
    query rows (see :class:`LayerCapture`); when ``True`` they are the full,
    unsliced block outputs and the caller is responsible for slicing after any
    gradient call.
    """
    with LayerCapture(model, split, keep_graph=keep_graph) as capture:
        if keep_graph:
            final = model.encode_table((x, y), split, num_mem_chunks=1)
        else:
            with torch.no_grad():
                final = model.encode_table((x, y), split, num_mem_chunks=num_mem_chunks)
    query_embedding = final[:, split:, -1, :]
    return query_embedding, capture.captures


def build_ensemble(
    model: NanoTabPFNModel,
    support_x: torch.Tensor,
    support_y: torch.Tensor,
    query_x: torch.Tensor,
    *,
    scheme: str,
    members: int,
    fraction: float = 0.8,
    seed: int = 2402,
    num_mem_chunks: int = 1,
    compute_gradient: bool = True,
    heldout_x: torch.Tensor | None = None,
) -> ResampleEnsemble:
    """Draw ``members`` resamples of ``support_{x,y}`` and score them on ``query_x``.

    All tensors are unbatched, ``(rows, features)`` / ``(rows,)`` / ``(query,
    features)`` -- one episode at a time, matching how the rest of the
    continuous-posterior experiments loop over episodes. The full-context
    base-point pass (for the reference embeddings and, if requested, the
    decoder-projection gradient) is computed here too, so callers get everything
    from one function.

    ``support_{x,y}`` should already be the *fitting* pool (see
    :func:`split_fit_and_heldout`); pass the held-out rows as ``heldout_x`` to
    score every member on them in the same batched forward pass. Held-out
    predictions are never used to build the layer captures or the decoder
    gradient, only recorded as ``heldout_member_probabilities``.
    """
    if scheme not in SCHEMES:
        raise ValueError(f"scheme must be one of {SCHEMES}, found {scheme!r}.")
    if members < 2:
        raise ValueError("members must be at least two.")
    fit_size = support_x.shape[0]
    generator = torch.Generator(device="cpu").manual_seed(seed)
    if scheme == "subsample":
        indices = _stratified_member_indices(support_y, members, fraction, seed)
    else:
        indices = _member_indices(scheme, fit_size, members, generator, fraction=fraction)
    device = support_x.device
    indices = indices.to(device)

    member_support_x = support_x[indices]
    member_support_y = support_y[indices]
    draw_size = member_support_x.shape[1]
    query_count = query_x.shape[0]
    expanded_query = query_x.unsqueeze(0).expand(members, -1, -1)
    scored_x = expanded_query if heldout_x is None else torch.cat(
        (expanded_query, heldout_x.unsqueeze(0).expand(members, -1, -1)), dim=1
    )
    stacked_x = torch.cat((member_support_x, scored_x), dim=1)

    scored_embedding, layer_captures = _encode_and_capture(
        model, stacked_x, member_support_y, draw_size, num_mem_chunks=num_mem_chunks, keep_graph=False
    )
    logits = model.decoder(scored_embedding)[..., :2]
    scored_probabilities = logits.softmax(dim=-1)[..., 1]
    member_probabilities, heldout_member_probabilities = (
        scored_probabilities[:, :query_count],
        scored_probabilities[:, query_count:] if heldout_x is not None else None,
    )
    layer_captures = [capture[:, :query_count, :] for capture in layer_captures]
    if scored_embedding.shape[1] != query_count + (0 if heldout_x is None else heldout_x.shape[0]):
        raise RuntimeError("Scored embedding count drifted from query+heldout; this should be impossible.")

    # The full-context base point: the reference embeddings for representation
    # dispersion, and -- only if requested -- the graph the decoder-projection
    # gradient is read from.
    base_x = torch.cat((support_x, query_x), dim=0).unsqueeze(0)
    base_y = support_y.unsqueeze(0)
    base_query_embedding, base_captures = _encode_and_capture(
        model, base_x, base_y, fit_size, num_mem_chunks=1, keep_graph=compute_gradient
    )
    with torch.no_grad():
        base_logits = model.decoder(base_query_embedding)[..., :2]
        base_positive = base_logits.softmax(dim=-1)[0, :, 1].detach()
    if compute_gradient:
        # base_captures are unsliced (batch, rows, cols, embedding); the query
        # slice is what belongs in the ensemble record.
        base_layer_embeddings = [capture[0, fit_size:, -1, :].detach().clone() for capture in base_captures]
    else:
        base_layer_embeddings = [capture[0].detach() for capture in base_captures]

    ensemble = ResampleEnsemble(
        scheme=scheme,
        member_probabilities=member_probabilities.detach(),
        layer_query_embeddings=[capture.detach() for capture in layer_captures],
        base_positive=base_positive,
        base_layer_embeddings=base_layer_embeddings,
        heldout_member_probabilities=(
            None if heldout_member_probabilities is None else heldout_member_probabilities.detach()
        ),
    )
    if compute_gradient:
        ensemble.decoder_gradient(model.decoder, base_captures, fit_size)
    return ensemble


# --------------------------------------------------------------------------- #
# Model-intrinsic arms (no resampling)
# --------------------------------------------------------------------------- #
@dataclass
class IntrinsicPosterior:
    """Joint mutual information and self-conditioning sensitivity, no resampling.

    Under exchangeability the aleatoric part of the predictive factorizes across
    queries given the latent function, so any residual dependence in the model's
    own *joint* over two queries is epistemic. This is read directly from the
    model by conditioning on a hypothetical label for one query and asking how
    much that moves belief about another: no ensemble, no resampling, and the
    estimator's target is the model's own posterior rather than an estimator's
    sensitivity to its conditioning set.

    The construction is a heuristic, not a proof of coherence: chaining
    ``p(y_q=a) * p(y_q'=b | D, y_q=a)`` is exactly the joint only if the model's
    one-step-ahead conditionals are consistent views of one true underlying
    joint, which an amortized predictor is not guaranteed to satisfy. Comparing
    this joint's mutual information against the *exact* candidate-posterior
    ground truth (:func:`exact_pairwise_mutual_information`) is exactly the
    check for whether that assumption costs anything in practice.

    Attributes:
        base_positive: ``(query,)`` unconditioned class-1 probability.
        conditional_positive: ``(query, 2, query)`` class-1 probability of
            query ``q'`` given the support extended with query ``q``'s row at
            hypothetical label ``a``: ``conditional_positive[q, a, q']``.
    """

    base_positive: torch.Tensor
    conditional_positive: torch.Tensor

    @property
    def query_count(self) -> int:
        return self.base_positive.shape[0]

    def self_conditioning(self) -> torch.Tensor:
        """``(query,)`` expected movement in a query's own belief under its two hypothetical labels."""
        diagonal = torch.diagonal(self.conditional_positive, dim1=0, dim2=2)  # (2, query)
        movement = (diagonal - self.base_positive[None, :]).abs()
        weight = torch.stack((1.0 - self.base_positive, self.base_positive))
        return (weight * movement).sum(dim=0)

    def joint_mutual_information(self) -> torch.Tensor:
        """``(query, query)`` pairwise ``I(y_q; y_q' | D)``, ``nan`` on the diagonal."""
        p_a = torch.stack((1.0 - self.base_positive, self.base_positive), dim=1)  # (query, 2)
        conditional = self.conditional_positive.clamp(1e-6, 1.0 - 1e-6)
        p_b_given_a = torch.stack((1.0 - conditional, conditional), dim=-1)  # (query, 2, query, 2)
        joint = p_a[:, :, None, None] * p_b_given_a  # joint[q, a, q', b]
        marginal_a = joint.sum(dim=-1)  # (query, 2, query) -- P(y_q = a), constant over q'
        marginal_b = joint.sum(dim=1)  # (query, query, 2) -- P(y_q' = b | conditioned on q)
        log_ratio = (
            joint.clamp_min(1e-12).log()
            - marginal_a.clamp_min(1e-12).log()[..., None]
            - marginal_b.clamp_min(1e-12).log()[:, None, :, :]
        )
        mutual_information = (joint * log_ratio).sum(dim=(1, 3)).clamp_min(0.0)
        mutual_information.fill_diagonal_(float("nan"))
        return mutual_information


def compute_intrinsic_posterior(
    model: NanoTabPFNModel,
    support_x: torch.Tensor,
    support_y: torch.Tensor,
    query_x: torch.Tensor,
    *,
    num_mem_chunks: int = 1,
) -> IntrinsicPosterior:
    """One batched forward pass over every ``(query, hypothetical label)`` pair.

    For each query ``q`` and hypothetical label ``a in {0, 1}``, the support is
    extended with ``(x_q, a)`` and every query (including ``q`` itself) is
    re-scored. This gives the full conditional table in ``2 * query_count``
    stacked forward passes worth of compute, batched into one call.
    """
    query_count = query_x.shape[0]
    support_size = support_x.shape[0]
    hypothetical_labels = torch.tensor([0.0, 1.0], device=support_x.device)
    # Member (q, a): support extended with query q's row at label a.
    extended_x = query_x[:, None, :].expand(-1, 2, -1).reshape(query_count * 2, 1, -1)
    extended_y = hypothetical_labels[None, :].expand(query_count, -1).reshape(query_count * 2, 1)
    member_support_x = torch.cat(
        (support_x[None].expand(query_count * 2, -1, -1), extended_x), dim=1
    )
    member_support_y = torch.cat((support_y[None].expand(query_count * 2, -1), extended_y), dim=1)
    member_query_x = query_x[None].expand(query_count * 2, -1, -1)
    stacked_x = torch.cat((member_support_x, member_query_x), dim=1)
    with torch.no_grad():
        final = model.encode_table(
            (stacked_x, member_support_y), support_size + 1, num_mem_chunks=num_mem_chunks
        )
        query_embedding = final[:, support_size + 1 :, -1, :]
        logits = model.decoder(query_embedding)[..., :2]
        conditional_positive = logits.softmax(dim=-1)[..., 1].reshape(query_count, 2, query_count)

        base_x = torch.cat((support_x, query_x), dim=0).unsqueeze(0)
        base_final = model.encode_table((base_x, support_y[None]), support_size, num_mem_chunks=1)
        base_logits = model.decoder(base_final[:, support_size:, -1, :])[..., :2]
        base_positive = base_logits.softmax(dim=-1)[0, :, 1]
    return IntrinsicPosterior(base_positive=base_positive.detach(), conditional_positive=conditional_positive.detach())


def exact_pairwise_mutual_information(
    candidate_query_positive: torch.Tensor, posterior: torch.Tensor
) -> torch.Tensor:
    """Ground truth ``(query, query)`` pairwise ``I(y_q; y_q' | D)`` under the exact candidate posterior.

    Labels are conditionally independent across queries given which candidate
    function is active (label noise is i.i.d. given the candidate), so the exact
    joint is a posterior-weighted mixture of independent Bernoulli products --
    no sampling and no learned model involved. This is what
    :meth:`IntrinsicPosterior.joint_mutual_information` and the resampling
    ensembles' pairwise structure are checked against.

    Args:
        candidate_query_positive: ``(candidate, query)`` noise-adjusted class-1
            probability of each candidate, e.g. one batch item of
            ``episode.candidate_query_positive``.
        posterior: ``(candidate,)`` exact posterior weight, e.g. one batch item
            of ``episode.posterior``.
    """
    theta = candidate_query_positive.clamp(1e-6, 1.0 - 1e-6)
    per_class = torch.stack((1.0 - theta, theta), dim=-1)  # (candidate, query, 2)
    joint = torch.einsum("c,cqa,crb->qrab", posterior, per_class, per_class)
    marginal_a = joint.sum(dim=-1)  # (query, query, 2) -- P(y_q = a), independent of r
    marginal_b = joint.sum(dim=2)  # (query, query, 2) -- P(y_r = b), independent of q
    log_ratio = (
        joint.clamp_min(1e-12).log()
        - marginal_a.clamp_min(1e-12).log()[:, :, :, None]
        - marginal_b.clamp_min(1e-12).log()[:, :, None, :]
    )
    mutual_information = (joint * log_ratio).sum(dim=(-1, -2)).clamp_min(0.0)
    mutual_information.fill_diagonal_(float("nan"))
    return mutual_information
