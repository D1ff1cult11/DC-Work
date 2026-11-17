import json, os

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_results")
dirs = sorted([d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d)) and os.path.exists(os.path.join(base, d, "test_metrics.json"))])

print(f"=== AUDITING ALL {len(dirs)} BENCHMARK RUNS ===")
for d in dirs:
    mf = os.path.join(base, d, "test_metrics.json")
    hf = os.path.join(base, d, "train_history.json")
    if os.path.exists(mf):
        m = json.load(open(mf))
        print(f"\n[{d}]")
        acc = m["accuracy"]*100
        mf1 = m["macro_f1"]*100
        mp = m["macro_precision"]*100
        mr = m["macro_recall"]*100
        kap = m["cohens_kappa"]
        fps = m.get("inference_fps", 0)
        params = m.get("parameters", 0)
        print(f"  Acc: {acc:5.2f}% | F1: {mf1:5.2f}% | Prec: {mp:5.2f}% | Rec: {mr:5.2f}% | Kappa: {kap:.4f} | FPS: {fps:6.1f} | Params: {params:,}")
        if os.path.exists(hf):
            h = json.load(open(hf))
            print(f"  Epochs: {len(h['train_loss'])} | Final TrLoss: {h['train_loss'][-1]:.4f} | Best Val F1: {max(h['val_f1'])*100:.2f}%")
