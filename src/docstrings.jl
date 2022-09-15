
const XGBOOST_DOCS_LINK = "https://xgboost.readthedocs.io/en/stable/index.html"
const XGBOOST_PARAMS_DOCS_LINK = "https://xgboost.readthedocs.io/en/stable/parameter.html"


"""
$(MMI.doc_header(XGBoostRegressor))

Univariate continuous regression using [xgboost]($XGBOOST_DOCS_LINK).

# Training data
In `MLJ` or `MLJBase`, bind an instance `model` to data with
```julia
m = machine(model, X, y)
```
where
- `X`: any table of input features, either an `AbstractMatrix` or Tables.jl-compatible table.
- `y`: is an `AbstractVector` continuous target.

Train using `fit!(m, rows=...)`.

# Hyper-parameters
See $XGBOOST_PARAMS_DOCS_LINK.
"""
XGBoostRegressor


"""
$(MMI.doc_header(XGBoostCount))

Univariate discrete regression using [xgboost]($XGBOOST_DOCS_LINK).

# Training data
In `MLJ` or `MLJBase`, bind an instance `model` to data with
```julia
m = machine(model, X, y)
```
where
- `X`: any table of input features, either an `AbstractMatrix` or Tables.jl-compatible table.
- `y`: is an `AbstractVector` continuous target.

Train using `fit!(m, rows=...)`.

# Hyper-parameters
See $XGBOOST_PARAMS_DOCS_LINK.
"""
XGBoostCount


"""
$(MMI.doc_header(XGBoostClassifier))

Univariate classification using [xgboost]($XGBOOST_DOCS_LINK).

# Training data
In `MLJ` or `MLJBase`, bind an instance `model` to data with
```julia
m = machine(model, X, y)
```
where
- `X`: any table of input features, either an `AbstractMatrix` or Tables.jl-compatible table.
- `y`: is an `AbstractVector` continuous target.

Train using `fit!(m, rows=...)`.

# Hyper-parameters
See $XGBOOST_PARAMS_DOCS_LINK.
"""
XGBoostClassifier


