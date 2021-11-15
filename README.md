# CatBoost
CatBoost (Categorical Boosting): unbiased boosting with categorical features

CatBoost is a relatively new open-source machine learning algorithm, developed in 2017 by Yandex. 
One of CatBoost’s core edges is its ability to integrate a variety of different data types, such as images, audio, or text features into one framework. But CatBoost also offers an idiosyncratic way of handling categorical data, requiring a minimum of categorical feature transformation, opposed to the majority of other machine learning algorithms, that cannot handle non-numeric values. From a feature engineering perspective, the transformation from a non-numeric state to numeric values can be a very non-trivial and tedious task, and CatBoost makes this step obsolete.

CatBoost builds upon the theory of decision trees and gradient boosting. The main idea of boosting is to sequentially combine many weak models (a model performing slightly better than random chance) and thus through greedy search create a strong competitive predictive model. Because gradient boosting fits the decision trees sequentially, the fitted trees will learn from the mistakes of former trees and hence reduce the errors. This process of adding a new function to existing ones is continued until the selected loss function is no longer minimized.

In the growing procedure of the decision trees, CatBoost does not follow similar gradient boosting models. Instead, CatBoost grows oblivious trees, which means that the trees are grown by imposing the rule that all nodes at the same level, test the same predictor with the same condition, and hence an index of a leaf can be calculated with bitwise operations. The oblivious tree procedure allows for a simple fitting scheme and efficiency on CPUs, while the tree structure operates as a regularization to find an optimal solution and avoid overfitting.

CatBoost still remains fairly unknown, but the algorithm offers immense flexibility with its approach to handling heterogeneous, sparse, and categorical data while still supporting fast training time and already optimized hyperparameters.

[Source](https://towardsdatascience.com/catboost-regression-in-6-minutes-3487f3e5b329)

## Installation

### Pip Install

```
!pip install catboost
``` 
For other methods [click](https://catboost.ai/en/docs/concepts/python-installation)

#### GPU system requirements
The versions of CatBoost available from pip install and conda install have GPU support out-of-the-box.

Devices with compute capability 3.0 and higher are supported in compiled packages.

Training on GPU requires NVIDIA Driver of version 418.xx or higher.

#### Dependencies:

numpy

six

pandas

## Parameters

```
class CatBoostRegressor(iterations=None,
                        learning_rate=None,
                        depth=None,
                        l2_leaf_reg=None,
                        model_size_reg=None,
                        rsm=None,
                        loss_function='RMSE',
                        border_count=None,
                        feature_border_type=None,
                        per_float_feature_quantization=None,
                        input_borders=None,
                        output_borders=None,
                        fold_permutation_block=None,
                        od_pval=None,
                        od_wait=None,
                        od_type=None,
                        nan_mode=None,
                        counter_calc_method=None,
                        leaf_estimation_iterations=None,
                        leaf_estimation_method=None,
                        thread_count=None,
                        random_seed=None,
                        use_best_model=None,
                        best_model_min_trees=None,
                        verbose=None,
                        silent=None,
                        logging_level=None,
                        metric_period=None,
                        ctr_leaf_count_limit=None,
                        store_all_simple_ctr=None,
                        max_ctr_complexity=None,
                        has_time=None,
                        allow_const_label=None,
                        one_hot_max_size=None,
                        random_strength=None,
                        name=None,
                        ignored_features=None,
                        train_dir=None,
                        custom_metric=None,
                        eval_metric=None,
                        bagging_temperature=None,
                        save_snapshot=None,
                        snapshot_file=None,
                        snapshot_interval=None,
                        fold_len_multiplier=None,
                        used_ram_limit=None,
                        gpu_ram_part=None,
                        pinned_memory_size=None,
                        allow_writing_files=None,
                        final_ctr_computation_mode=None,
                        approx_on_full_history=None,
                        boosting_type=None,
                        simple_ctr=None,
                        combinations_ctr=None,
                        per_feature_ctr=None,
                        ctr_target_border_count=None,
                        task_type=None,
                        device_config=None,                        
                        devices=None,
                        bootstrap_type=None,
                        subsample=None,                        
                        sampling_unit=None,
                        dev_score_calc_obj_block_size=None,
                        max_depth=None,
                        n_estimators=None,
                        num_boost_round=None,
                        num_trees=None,
                        colsample_bylevel=None,
                        random_state=None,
                        reg_lambda=None,
                        objective=None,
                        eta=None,
                        max_bin=None,
                        gpu_cat_features_storage=None,
                        data_partition=None,
                        metadata=None,
                        early_stopping_rounds=None,
                        cat_features=None,
                        grow_policy=None,
                        min_data_in_leaf=None,
                        min_child_samples=None,
                        max_leaves=None,
                        num_leaves=None,
                        score_function=None,
                        leaf_estimation_backtracking=None,
                        ctr_history_unit=None,
                        monotone_constraints=None,
                        feature_weights=None,
                        penalties_coefficient=None,
                        first_feature_use_penalties=None,
                        model_shrink_rate=None,
                        model_shrink_mode=None,
                        langevin=None,
                        diffusion_temperature=None,
                        posterior_sampling=None,
                        boost_from_average=None)
```
## Dataset: Credit Card Data from book "Econometric Analysis"

Context

A small credit card dataset for simple econometric analysis,

Content

- card: Dummy variable, 1 if application for credit card accepted, 0 if not
- reports: Number of major derogatory reports
- age: Age n years plus twelfths of a year
- income: Yearly income (divided by 10,000)
- share: Ratio of monthly credit card expenditure to yearly income
- expenditure: Average monthly credit card expenditure
- owner: 1 if owns their home, 0 if rent
- selfempl: 1 if self employed, 0 if not.
- dependents: 1 + number of dependents
- months: Months living at current address
- majorcards: Number of major credit cards held
- active: Number of active credit accounts

Acknowledgements

This dataset was originally published alongside the 5th edition of William Greene's book Econometric Analysis.

## Example

- Dataset: AER_credit_card_data.csv
 - Preprocessing: 
   - Cleaning - missing data
   - Normalization - Label encoder
 - Evaluation Metric: RMSE
 - Hiperparameter Optimization: GridSearchCV

