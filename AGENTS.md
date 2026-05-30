# Agent Guidelines: proto-counterfactuals

## ⚙️ Setup & Dependencies
*   **Environment Manager:** Use `pixi` for dependency management. It reads configurations from `pixi.toml`.
*   **PyTorch Specifics:** Note the platform-specific dependencies outlined in `pixi.toml` (e.g., use the correct index for CUDA versions or fallback to `cpu` for OSX).

## 📁 Directory Structure
The codebase is highly modular, dedicated to specific tasks:
*   `src/data_preprocessing/`: Contains scripts responsible for splitting and preparing raw data (e.g., `*.py` files within `ph2/`, `papila/`, `cub2002011/`).
*   `src/protopnet/`: Contains the core ProtoPNet model and training logic.
*   `src/deformable-protopnet/`: Contains the deformable ProtoPNet implementation and associated utilities.
*   `src/baseline/`: Dedicated to model testing against various datasets.

## 🧪 Testing & Execution Flow
*   **Data Splitting:** Always respect the `train_size`, `val_size`, and `test_size` arguments (which must sum to 1.0) used in the dedicated `split_train_test.py` scripts. These scripts use `StratifiedShuffleSplit` to ensure consistent class distribution across splits.
    *   *Example*: See `src/data_preprocessing/ph2/split_train_test.py` for implementation details.
*   **Validation Check:** When training or testing, the metrics collected and returned should include: `loss_train`, `loss_val`, and `run_avg_loss` (according to `src/protopnet/models_train.py:525`).
*   **Model Testing:** Testing generally involves calling a dedicated `model_test` function with a loaded dataset/loader for test, validation, and training splits (e.g., `src/baseline/models_test.py:228`).

## 🚀 General Workflow Quirks
*   **Preferred Source of Truth:** Trust executable code (scripts/configs) over prose/documentation if a conflict arises.
*   **Required Command Order:** Follow prescribed order when performing a full pipeline (e.g., `[Data Prep] -> [Train] -> [Test]`).
*   **Debugging:** Debug tests often involve using `matplotlib.pyplot.imread` (as seen in `src/deformable-protopnet/data_utilities.py:99`).