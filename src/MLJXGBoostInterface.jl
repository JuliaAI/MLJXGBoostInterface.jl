module MLJXGBoostInterface

export XGBoostRegressor, XGBoostClassifier, XGBoostCount

import MLJModelInterface
using MLJModelInterface: Table, Continuous, Count, Finite, OrderedFactor, Multiclass

const PKG = "MLJXGBoostInterface"
const MMI = MLJModelInterface

using Base: @kwdef
using Tables: schema

import XGBoost
import SparseArrays

const Maybe{T} = Union{Nothing,T}

# helper for feature importances:
# XGBoost used "f
function named_importance(fi::XGBoost.FeatureImportance, features)

    new_fname = features[parse(Int, fi.fname[2:end]) + 1] |> string
    return XGBoost.FeatureImportance(new_fname, fi.gain, fi.cover, fi.freq)
end

abstract type XGBoostAbstractRegressor <: MMI.Deterministic end
abstract type XGBoostAbstractClassifier <: MMI.Probabilistic end


## REGRESSOR

"""
    XGBoostRegressor(; objective="linear", seed=0, kwargs...)

The XGBoost model for univariate targets with `Continuous` element
scitype. Gives deterministic (point) predictions. For possible values
for `objective` and `kwargs`, see
[https://xgboost.readthedocs.io/en/latest/parameter.html](https://xgboost.readthedocs.io/en/latest/parameter.html).

For a time-dependent random seed, use `seed=-1`.

See also: XGBoostCount, XGBoostClassifier

"""
@kwdef mutable struct XGBoostRegressor <: XGBoostAbstractRegressor
    num_round::Int = 100  # this must be provided since XGBoost.jl doesn't provide it
    booster::Maybe{String} = nothing
    disable_default_eval_metric::Maybe{Int} = nothing
    eta::Maybe{Float64} = nothing
    num_parallel_tree::Maybe{Int} = nothing
    gamma::Maybe{Float64} = nothing
    max_depth::Maybe{Int} = nothing
    min_child_weight::Maybe{Float64} = nothing
    max_delta_step::Maybe{Float64} = nothing
    subsample::Maybe{Float64} = nothing
    colsample_bytree::Maybe{Float64} = nothing
    colsample_bylevel::Maybe{Float64} = nothing
    colsample_bynode::Maybe{Float64} = nothing
    lambda::Maybe{Float64} = nothing
    alpha::Maybe{Float64} = nothing
    tree_method::Maybe{String} = nothing
    sketch_eps::Maybe{Float64} = nothing
    scale_pos_weight::Maybe{Float64} = nothing
    updater::Maybe{Float64} = nothing
    refresh_leaf::Maybe{Union{Int,Bool}} = 1
    process_type::Maybe{String} = nothing
    grow_policy::Maybe{String} = nothing
    max_leaves::Maybe{Int} = nothing
    max_bin::Maybe{Int} = nothing
    predictor::Maybe{String} = "cpu_predictor"
    sample_type::Maybe{String} = nothing
    normalize_type::Maybe{String} = nothing
    rate_drop::Maybe{Float64} = nothing
    one_drop::Maybe{Union{Int,Bool}} = nothing
    skip_drop::Maybe{Float64} = nothing
    feature_selector::Maybe{String} = nothing
    top_k::Maybe{Int} = nothing
    tweedie_variance_power::Maybe{Float64} = nothing
    objective = "reg:squarederror"
    base_score::Maybe{Float64} = nothing
    eval_metric = "rmse"
    seed::Maybe{Int} = nothing
    nthread::Maybe{Int} = nothing

    XGBoostRegressor(a...) = _constructor_checks(new(a...))
end

function _fix_objective(obj)
    obj ∈ ("linear", "gamma", "tweedie") ? "reg:"*obj : obj
end

function kwargs(model, verbosity::Integer, obj, eval=nothing)
    fn = fieldnames(typeof(model))
    o = NamedTuple(n=>getfield(model, n) for n ∈ fn if !isnothing(getfield(model, n)))
    o = merge(o, (silent=(verbosity == 0),))
    isnothing(eval) || (o = merge(o, (eval=eval,)))
    merge(o, (objective=_fix_objective(obj),))
end

function MMI.clean!(model::XGBoostRegressor)
    warning = ""
    if model.objective == "count:poisson"
        warning *= "Your `objective` suggests prediction of a "*
        "`Count` variable.\n You may want to consider XGBoostCount instead. "
    elseif model.objective in ["reg:logistic", "binary:logistic",
                               "binary:logitraw", "binary:hinge",
                               "multi:softmax", "multi:softprob"]
        warning *="Your `objective` suggests prediction of a "*
        "`Finite` variable.\n You may want to consider XGBoostClassifier "*
        "instead. "
    end
    return warning
end

function _constructor_checks(model)
    msg = MMI.clean(model)
    isempty(msg) || @warn msg
    model
end

# For `XGBoost.DMatrix(Xmatrix, y)` `Xmatrix` must either be a julia `Array` or
# a `SparseMatrixCSC` while `y` must be a `Vector` 
_to_array(x::Union{Array, SparseArrays.SparseMatrixCSC}) = x
_to_array(x::AbstractArray) = copyto!(similar(Array{eltype(x)}, axes(x)), x) 

function importances(X, r)
    fs = schema(X).names
    [named_importance(fi, fs) for fi ∈ XGBoost.importance(r)]
end

function MMI.fit(model::XGBoostAbstractRegressor, verbosity::Integer, X, y)
    Xmatrix = _to_array(MMI.matrix(X))
    dm = DMatrix(Xmatrix, label=_to_array(y))

    r = xgboost(dm, model.num_round; kwargs(model, verbosity, obj)...)

    report = (feature_importances=importances(X, r), )

    (r, nothing, report)
end

function MMI.predict(model::XGBoostAbstractRegressor, fitresult, Xnew)
    Xmatrix = _to_array(MMI.matrix(Xnew))
    return XGBoost.predict(fitresult, Xmatrix)
end


"""
    XGBoostCount(; seed=0, kwargs...)

The XGBoost model for targets with `Count` scitype. Gives
deterministic (point) predictions. For admissible `kwargs`, see
[https://xgboost.readthedocs.io/en/latest/parameter.html](https://xgboost.readthedocs.io/en/latest/parameter.html).

For a time-dependent random seed, use `seed=-1`.

See also: XGBoostRegressor, XGBoostClassifier

"""
@kwdef mutable struct XGBoostCount <:MMI.Deterministic
    num_round::Int = 100  # this must be provided since XGBoost.jl doesn't provide it
    booster::Maybe{String} = nothing
    disable_default_eval_metric::Maybe{Int} = nothing
    eta::Maybe{Float64} = nothing
    num_parallel_tree::Maybe{Int} = nothing
    gamma::Maybe{Float64} = nothing
    max_depth::Maybe{Int} = nothing
    min_child_weight::Maybe{Float64} = nothing
    max_delta_step::Maybe{Float64} = nothing
    subsample::Maybe{Float64} = nothing
    colsample_bytree::Maybe{Float64} = nothing
    colsample_bylevel::Maybe{Float64} = nothing
    colsample_bynode::Maybe{Float64} = nothing
    lambda::Maybe{Float64} = nothing
    alpha::Maybe{Float64} = nothing
    tree_method::Maybe{String} = nothing
    sketch_eps::Maybe{Float64} = nothing
    scale_pos_weight::Maybe{Float64} = nothing
    updater::Maybe{Float64} = nothing
    refresh_leaf::Maybe{Union{Int,Bool}} = 1
    process_type::Maybe{String} = nothing
    grow_policy::Maybe{String} = nothing
    max_leaves::Maybe{Int} = nothing
    max_bin::Maybe{Int} = nothing
    predictor::Maybe{String} = "cpu_predictor"
    sample_type::Maybe{String} = nothing
    normalize_type::Maybe{String} = nothing
    rate_drop::Maybe{Float64} = nothing
    one_drop::Maybe{Union{Int,Bool}} = nothing
    skip_drop::Maybe{Float64} = nothing
    feature_selector::Maybe{String} = nothing
    top_k::Maybe{Int} = nothing
    tweedie_variance_power::Maybe{Float64} = nothing
    objective = "count:poisson"
    base_score::Maybe{Float64} = nothing
    eval_metric = "rmse"
    seed::Maybe{Int} = nothing
    nthread::Maybe{Int} = nothing

    XGBoostCount(a...) = _constructor_checks(new(a...))
end

function MMI.clean!(model::XGBoostCount)
    warning = ""
    if(!(model.objective in ["count:poisson"]))
        warning *= "Changing objective to \"poisson\", "*
        "the only supported value. "
        model.objective="poisson"
    end
    return warning
end

"""
    XGBoostClassifier(; seed=0, kwargs...)

The XGBoost model for targets with `Finite` scitype (which includes
`Binary=Finite{2}`). Gives probabilistic predictions. For admissible
`kwargs`, see
[https://xgboost.readthedocs.io/en/latest/parameter.html](https://xgboost.readthedocs.io/en/latest/parameter.html).

For a time-dependent random seed, use `seed=-1`.

See also: XGBoostCount, XGBoostRegressor

"""
@kwdef mutable struct XGBoostClassifier <:MMI.Probabilistic
    num_round::Int = 100  # this must be provided since XGBoost.jl doesn't provide it
    booster::Maybe{String} = nothing
    disable_default_eval_metric::Maybe{Int} = nothing
    eta::Maybe{Float64} = nothing
    num_parallel_tree::Maybe{Int} = nothing
    gamma::Maybe{Float64} = nothing
    max_depth::Maybe{Int} = nothing
    min_child_weight::Maybe{Float64} = nothing
    max_delta_step::Maybe{Float64} = nothing
    subsample::Maybe{Float64} = nothing
    colsample_bytree::Maybe{Float64} = nothing
    colsample_bylevel::Maybe{Float64} = nothing
    colsample_bynode::Maybe{Float64} = nothing
    lambda::Maybe{Float64} = nothing
    alpha::Maybe{Float64} = nothing
    tree_method::Maybe{String} = nothing
    sketch_eps::Maybe{Float64} = nothing
    scale_pos_weight::Maybe{Float64} = nothing
    updater::Maybe{Float64} = nothing
    refresh_leaf::Maybe{Union{Int,Bool}} = 1
    process_type::Maybe{String} = nothing
    grow_policy::Maybe{String} = nothing
    max_leaves::Maybe{Int} = nothing
    max_bin::Maybe{Int} = nothing
    predictor::Maybe{String} = "cpu_predictor"
    sample_type::Maybe{String} = nothing
    normalize_type::Maybe{String} = nothing
    rate_drop::Maybe{Float64} = nothing
    one_drop::Maybe{Union{Int,Bool}} = nothing
    skip_drop::Maybe{Float64} = nothing
    feature_selector::Maybe{String} = nothing
    top_k::Maybe{Int} = nothing
    tweedie_variance_power::Maybe{Float64} = nothing
    objective = "automatic"
    base_score::Maybe{Float64} = nothing
    eval_metric = "mlogloss"
    seed::Maybe{Int} = nothing
    nthread::Maybe{Int} = nothing

    XGBoostClassifier(a...) = _constructor_checks(new(a...))
end

function MMI.clean!(model::XGBoostClassifier)
    warning = ""
    if(!(model.objective =="automatic"))
        warning *="Changing objective to \"automatic\", the only supported value. "
        model.objective="automatic"
    end
    return warning
end

function MMI.fit(model::XGBoostClassifier
                 , verbosity::Int     #> must be here even if unsupported in pkg
                 , X
                 , y)
    Xmatrix = _to_array(MMI.matrix(X))

    a_target_element = y[1] # a CategoricalValue or CategoricalString
    nclass = length(MMI.classes(a_target_element))

    # The eval metric is different depending on the class size.
    eval_metric = model.eval_metric
    if nclass == 2 && eval_metric == "mlogloss"
        eval_metric = "logloss"
    end
    if nclass > 2 && eval_metric == "logloss"
        eval_metric = "mlogloss"
    end

    y_plain_ = MMI.int(y) .- 1 # integer relabeling should start at 0

    if nclass == 2
        objective = "binary:logistic"
        y_plain_ = convert(Array{Bool}, y_plain_)
    else
        objective = "multi:softprob"
    end

    y_plain = _to_array(y_plain_)

    nclass_arg = nclass == 2 ? (;) : (num_class=nclass,)

    r = xgboost(Xmatrix, label=y_plain, model.num_round;
                nclass_arg...,
                kwargs(model, verbosity, objective, eval_metric)...
               )
    fr = (result, a_target_element)

    report = (feature_importances=importances(X, r), )

    (fr, nothing, report)
end


function MMI.predict(model::XGBoostClassifier, fitresult, Xnew)
    (result, a_target_element) = fitresult
    decode = MMI.decoder(a_target_element)
    classes = MMI.classes(a_target_element)

    Xmatrix = _to_array(MMI.matrix(Xnew))
    XGBpredictions = XGBoost.predict(result, Xmatrix)

    nlevels = length(classes)
    npatterns = MMI.nrows(Xnew)

    if nlevels == 2
        true_class_probabilities = reshape(XGBpredictions, 1, npatterns)
        false_class_probabilities = 1 .- true_class_probabilities
        XGBpredictions = vcat(false_class_probabilities, true_class_probabilities)
    end

    prediction_probabilities = reshape(XGBpredictions, nlevels, npatterns)

    # note we use adjoint of above:
    MMI.UnivariateFinite(classes, prediction_probabilities')
end


# # SERIALIZATION


# ## Helpers

"""
    persistent(booster)

Private method.

Return a persistent (ie, Julia-serializable) representation of the
XGBoost.jl model `booster`.

Restore the model with [`booster`](@ref)

"""
function persistent(booster)

    # this implemenation is not ideal; see
    # https://github.com/dmlc/XGBoost.jl/issues/103

    xgb_file, io = mktemp()
    close(io)

    XGBoost.save(booster, xgb_file)
    persistent_booster = read(xgb_file)
    rm(xgb_file)
    return persistent_booster
end

"""
    booster(persistent)

Private method.

Return the XGBoost.jl model which has `persistent` as its persistent
(Julia-serializable) representation. See [`persistent`](@ref) method.

"""
function booster(persistent)

    xgb_file, io = mktemp()
    write(io, persistent)
    close(io)
    booster = XGBoost.Booster(model_file=xgb_file)
    rm(xgb_file)

    return booster
end


# ## Regressor and Count

const XGBoostInfinite = Union{XGBoostRegressor,XGBoostCount}

MLJModelInterface.save(::XGBoostInfinite, fitresult; kwargs...) =
persistent(fitresult)

MLJModelInterface.restore(::XGBoostInfinite, serializable_fitresult) =
booster(serializable_fitresult)


# ## Classifier

function MLJModelInterface.save(::XGBoostClassifier,
                                fitresult;
                                kwargs...)
    booster, a_target_element = fitresult
    return (persistent(booster), a_target_element)
end

function MLJModelInterface.restore(::XGBoostClassifier,
                                   serializable_fitresult)
    persistent, a_target_element = serializable_fitresult
    return (booster(persistent), a_target_element)
end


## METADATA

XGTypes=Union{XGBoostRegressor,XGBoostCount,XGBoostClassifier}

MMI.package_name(::Type{<:XGTypes}) = "XGBoost"
MMI.package_uuid(::Type{<:XGTypes}) = "009559a3-9522-5dbb-924b-0b6ed2b22bb9"
MMI.package_url(::Type{<:XGTypes}) = "https://github.com/dmlc/XGBoost.jl"
MMI.is_pure_julia(::Type{<:XGTypes}) = false

MMI.load_path(::Type{<:XGBoostRegressor}) = "$PKG.XGBoostRegressor"
MMI.input_scitype(::Type{<:XGBoostRegressor}) = Table(Continuous)
MMI.target_scitype(::Type{<:XGBoostRegressor}) = AbstractVector{Continuous}
MMI.docstring(::Type{<:XGBoostRegressor}) =
"The XGBoost gradient boosting method, for use with "*
"`Continuous` univariate targets. "

MMI.load_path(::Type{<:XGBoostCount}) = "$PKG.XGBoostCount"
MMI.input_scitype(::Type{<:XGBoostCount}) = Table(Continuous)
MMI.target_scitype(::Type{<:XGBoostCount}) = AbstractVector{Count}
MMI.docstring(::Type{<:XGBoostCount}) =
"The XGBoost gradient boosting method, for use with "*
"`Count` univariate targets, using a Poisson objective function. "

MMI.load_path(::Type{<:XGBoostClassifier}) = "$PKG.XGBoostClassifier"
MMI.input_scitype(::Type{<:XGBoostClassifier}) = Table(Continuous)
MMI.target_scitype(::Type{<:XGBoostClassifier}) = AbstractVector{<:Finite}
MMI.docstring(::Type{<:XGBoostClassifier}) =
"The XGBoost gradient boosting method, for use with "*
"`Finite` univariate targets (`Multiclass`, "*
"`OrderedFactor` and `Binary=Finite{2}`)."

end # module
