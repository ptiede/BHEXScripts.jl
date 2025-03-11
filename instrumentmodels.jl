
@inline fgain(x) = @fastmath exp(x.lg + 1im*x.gp)

@inline function gain(x)
    gR = exp(x.lgR + 1im*x.gpR)
    lgrat = x.lgratμ + x.lgratσ*x.lgrat
    gprat = x.gprat
    gL =gR*exp(lgrat + 1im*gprat)
    return gR, gL
end

@inline jfr(g, d, r) = adjoint(r)*g*d*r
@inline jnofr(g, d, r) = g*d*r

@inline function dterm(x)
    dR = complex(x.dRre, x.dRim)
    dL = complex(x.dLre, x.dLim)
    return dR, dL
end


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

function build_instrument_circular(;refsite=:ALMA, frcal=true, lgamp_sigma=0.2)

    G = JonesG(gain)

    D = JonesD(dterm)

    R = JonesR(;add_fr=true)

    if frcal
        J = JonesSandwich(jfr, G, D, R)
    else
        J = JonesSandwich(jnofr, G, D, R)
    end

    intprior = (
        lgR   = ArrayPrior(IIDSitePrior(IntegSeg(), Normal(0.0, lgamp_sigma)); gain_amp_override...),
        lgrat = ArrayPrior(IIDSitePrior(IntegSeg(), Normal(0.0, 1.0))),
        lgratμ= ArrayPrior(IIDSitePrior(TrackSeg(), Normal(0.0, 0.2))),
        lgratσ= ArrayPrior(IIDSitePrior(TrackSeg(), Exponential(0.2))),
        gpR   = ArrayPrior(IIDSitePrior(IntegSeg(), DiagonalVonMises(0.0, inv(π^2))); refant=SEFDReference(0.0), phase=true),
        gprat = ArrayPrior(IIDSitePrior(IntegSeg(), DiagonalVonMises(0.0, inv(0.5^2))); refant=SingleReference(refsite, 0.0), phase=true),
        dRre  = ArrayPrior(IIDSitePrior(TrackSeg(), Normal(0.0, 0.15))),
        dRim  = ArrayPrior(IIDSitePrior(TrackSeg(), Normal(0.0, 0.15))),
        dLre  = ArrayPrior(IIDSitePrior(TrackSeg(), Normal(0.0, 0.15))),
        dLim  = ArrayPrior(IIDSitePrior(TrackSeg(), Normal(0.0, 0.15))),
    )

    return InstrumentModel(J, intprior)
end
