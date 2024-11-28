
@inline fgain(x) = @fastmath exp(x.lg + 1im*x.gp)

function build_instrument(;lgamp_sigma = 0.2, refant=SEFDReference(0.0))
    G = SingleStokesGain(fgain)
    intprior = (
        lg = ArrayPrior(IIDSitePrior(IntegSeg(), Normal(0.0, lgamp_sigma));),
        gp = ArrayPrior(IIDSitePrior(IntegSeg(), DiagonalVonMises(0.0, inv(π^2))), refant=refant, phase=true;
                        space = IIDSitePrior(IntegSeg(), DiagonalVonMises(0.0, inv(π^2)))
                        )
    )

    return InstrumentModel(G, intprior)
end