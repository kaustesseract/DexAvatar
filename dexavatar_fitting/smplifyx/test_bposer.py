import os, glob


def expid2model(expr_dir):
    from configer import Configer

    if not os.path.exists(expr_dir): raise ValueError('Could not find the experiment directory: %s' % expr_dir)

    best_model_fname = sorted(glob.glob(os.path.join(expr_dir, 'snapshots', '*.pt')), key=os.path.getmtime)[-1]
    try_num = os.path.basename(best_model_fname).split('_')[0]

    print(('Found Trained Model: %s' % best_model_fname))

    default_ps_fname = glob.glob(os.path.join(expr_dir,'*.ini'))[0]
    if not os.path.exists(
        default_ps_fname): raise ValueError('Could not find the appropriate signbposer_settings: %s' % default_ps_fname)
    ps = Configer(default_ps_fname=default_ps_fname, work_dir = expr_dir, best_model_fname=best_model_fname)

    return ps, best_model_fname

def load_signbposer(expr_dir, sbp_model='snapshot'):

    import importlib
    import os
    import torch

    ps, trained_model_fname = expid2model(expr_dir)
    if sbp_model == 'snapshot':

        signbposer_path = sorted(glob.glob(os.path.join(expr_dir, 'signbposer.py')), key=os.path.getmtime)[-1]

        spec = importlib.util.spec_from_file_location('SignbPoser', signbposer_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        signbposer_pt = getattr(module, 'SignbPoser')(num_neurons=ps.num_neurons, latentD=ps.latentD, data_shape=ps.data_shape)
    else:
        signbposer_pt = sbp_model(num_neurons=ps.num_neurons, latentD=ps.latentD, data_shape=ps.data_shape)

    signbposer_pt.load_state_dict(torch.load(trained_model_fname, map_location='cpu'))
    signbposer_pt.eval()

    return signbposer_pt, ps

