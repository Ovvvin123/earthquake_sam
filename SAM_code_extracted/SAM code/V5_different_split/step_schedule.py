"""Schedules use completed optimizer updates, not epochs or microbatches."""
def online_weight(step, args):
    if not args.onlineCorrection or step < args.onlineStartStep:
        return 0.
    if step <= args.onlineHoldStep:
        return args.onlineWeightStart
    frac = min(1., (step-args.onlineHoldStep)/(args.onlineRampStep-args.onlineHoldStep))
    return args.onlineWeightStart + frac*(args.onlineWeightMax-args.onlineWeightStart)

def consistency_weight(step, args):
    if not args.unimatch or step < args.unimatchStartStep:
        return 0.
    return args.unimatchWeight*min(1., (step-args.unimatchStartStep+1)/
                                  (args.unimatchRampStep-args.unimatchStartStep+1))

def lr_factor(step, warmup):
    # Reproduce the old early milestones in optimizer-update units (20 updates/old epoch).
    warm = min(1., (step+1)/max(warmup, 1))
    decay = 1. if step < 40 else .5 if step < 80 else .25 if step < 160 else .125
    return warm*decay
