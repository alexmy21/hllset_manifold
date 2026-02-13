import pytest

from core.manifold_os import NTokenRepresentation, LUTRecord
from core.hllset import HLLSet


def test_disambiguate_simple():
    # tokens: ['a', 'b'] -> build 1-token groups and LUTs
    tokens = ['a', 'b']
    rep = NTokenRepresentation(original_tokens=tokens)
    rep.build_n_token_groups([1], maintain_order=True)

    # compute reg/zeros for each token using default seed
    pairs = HLLSet.compute_reg_zeros_batch(tokens)

    # build simple LUT for n=1
    lut = {}
    for tok, pair in zip(tokens, pairs):
        reg, zeros = pair
        rec = LUTRecord(reg=reg, zeros=zeros)
        # add token sequence as single-token tuple
        rec.add_entry(hash(tok), (tok,))
        lut[(reg, zeros)] = rec

    rep.luts[1] = lut

    # test disambiguation for token 'a'
    reg_a, zeros_a = pairs[0]
    candidates = rep.disambiguate_tokens(reg_a, zeros_a)
    assert 'a' in candidates
