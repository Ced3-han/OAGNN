import pandas as pd
import io
import PIL
from matplotlib import pyplot as plt
from torchvision.transforms import ToTensor

from utils.misc import BlackHole


def report_correlations(results, logger=BlackHole(), writer=BlackHole, it=0):
    per_target = []
    for key, val in results.groupby(['target']):
        # Ignore target with 2 decoys only since the correlations are
        # not really meaningful.
        if val.shape[0] < 3:
            continue
        true = val['true'].astype(float)
        pred = val['pred'].astype(float)
        pearson = true.corr(pred, method='pearson')
        kendall = true.corr(pred, method='kendall')
        spearman = true.corr(pred, method='spearman')
        per_target.append((key, pearson, kendall, spearman))
    per_target = pd.DataFrame(
        data=per_target,
        columns=['target', 'pearson', 'kendall', 'spearman'])

    res = {}
    all_true = results['true'].astype(float)
    all_pred = results['pred'].astype(float)
    res['all_pearson'] = all_true.corr(all_pred, method='pearson')
    res['all_kendall'] = all_true.corr(all_pred, method='kendall')
    res['all_spearman'] = all_true.corr(all_pred, method='spearman')

    res['per_target_pearson'] = per_target['pearson'].mean()
    res['per_target_kendall'] = per_target['kendall'].mean()
    res['per_target_spearman'] = per_target['spearman'].mean()

    logger.info(
        '[Iteration {:d}]\nCorrelations (Pearson, Kendall, Spearman)\n'
        '    per-target: ({:.3f}, {:.3f}, {:.3f})\n'
        '    global    : ({:.3f}, {:.3f}, {:.3f})'.format(
        it,
        float(res["per_target_pearson"]),
        float(res["per_target_kendall"]),
        float(res["per_target_spearman"]),
        float(res["all_pearson"]),
        float(res["all_kendall"]),
        float(res["all_spearman"]))
    )

    writer.add_scalar('val/per_target_pearson', res["per_target_pearson"], it)
    writer.add_scalar('val/per_target_kendall', res["per_target_kendall"], it)
    writer.add_scalar('val/per_target_spearman', res["per_target_spearman"], it)
    writer.add_scalar('val/all_pearson', res["all_pearson"], it)
    writer.add_scalar('val/all_kendall', res["all_kendall"], it)
    writer.add_scalar('val/all_spearman', res["all_spearman"], it)

    writer.add_image('val/scatter', plot_corr(all_true, all_pred), it)

    return res


def plot_corr(y_true, y_pred):
    plt.figure(figsize=(5,5))
    plt.scatter(y_true, y_pred, alpha=0.1)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = PIL.Image.open(buf)
    image = ToTensor()(image)
    plt.close()
    return image
