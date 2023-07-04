import os
import ray
from ray import tune, air
from ray.air import session
from ray.air.checkpoint import Checkpoint
from ray.tune.schedulers import ASHAScheduler
import shutil
import tempfile
from pathlib import Path

from guppyocr.predict_pxml import GuppyPageXMLPredictor
from libpagexml3.libpagexml3 import PageXMLPage, PageXMLTextTypes
import editdistance
from ray.tune.search.optuna import OptunaSearch


TRAIN_DATASET = os.environ.get("TRAIN_DS") or "/tmp/train_ds.pq"
VAL_DATASET = os.environ.get("VAL_DS") or ""
EXP_NAME = os.environ.get("EXP_NAME") or "guppy_experiment"

PXML_TEST_DATASETS = ["/tmp/TestDS"]
OUTPUT_PATH=""



def eval_pagexml(p: PageXMLPage) -> float:
    total_chars = 0
    total_errs = 0
    for tl in p.text_lines():
        try:
            if tl.has_pred_text() and tl.has_gt_text() and p.region_by_id(tl.parent_region_id).region_text_type not in {PageXMLTextTypes.DROP_CAPITAL, PageXMLTextTypes.CAPITAL_WORD, PageXMLTextTypes}:
                distance = editdistance.distance(tl.get_text_gt(), tl.get_text_pred())
                total_chars += max(len(tl.get_text_gt()), len(tl.get_text_pred()))
                total_errors += distance
        except:
            pass
    return total_chars, total_errs


def objective(config):
    # make a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        from guppyocr.train_calamares import TrainingOpts, train_model
        opts = TrainingOpts(output=tmpdir,dataset=str(TRAIN_DATASET),val_dataset=str(VAL_DATASET),reader="pq",arch="crnn",gpu=True,lr=config["lr"],optimizer=config["optimizer"], batch_size=config["batch_size"], early_stopping=3, epoch=30)
        trained_model_path = train_model(opts)
        if trained_model_path.exists():
            pass
            #shutil.copy2(trained_model_path, model_output_path)
            # TODO: Write model to OUTPUT_PATH Using a Unique name

        # load the model
        predictor = GuppyPageXMLPredictor(trained_model_path,dry_run=True)

        # evaluate the model
        cer_s = []
        for test_dataset in PXML_TEST_DATASETS:
            total_chars = 0
            total_erros = 0
            for f in Path(test_dataset).iterdir():
                if f.suffix == ".xml":
                    page = predictor.predict_pagexml(f)
                    chars, errs = eval_pagexml(page)
                    total_chars += chars
                    total_erros += errs
            if total_chars > 0:
                cer_s += total_erros / total_chars
        session.report({"mean_cer":sum(cer_s) / max(1,len(cer_s))})

def main(num_samples=10, max_num_epochs=10, gpus_per_trial=2):
    search_space = {
        "lr": tune.loguniform(1e-5, 1e-3),
        "batch_size": tune.choice([2, 4, 8, 16, 32, 48]),
        "optimizer": tune.choice(["SGD","Adam", "AdaBelief"]),
        "arch": tune.choice(["crnn","calamari"])
    }

    algo = OptunaSearch()

    tuner = tune.Tuner(
        tune.with_resources(objective,resources={"gpu":1}),
        tune_config=tune.TuneConfig(
            metric="mean_cer",
            mode="min",
            search_alg=algo,max_concurrent_trials=1
        ),
        run_config=air.RunConfig(
            stop={"training_iteration": 5},
            name=EXP_NAME
        ),
        param_space=search_space,
    )
    results = tuner.fit()
    print("Best config is:", results.get_best_result().config)
    print("Results:")
    print(results)

if __name__ == "__main__":
    main(num_samples=2, max_num_epochs=2, gpus_per_trial=0)
