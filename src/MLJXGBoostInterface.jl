module MLJXGBoostInterface

import MLJModelInterface; const MMI = MLJModelInterface
using MLJModelInterface: Table, Continuous, Count, Finite, OrderedFactor, Multiclass

const PKG = "MLJXGBoostInterface"

import Tables
using Tables: schema

import XGBoost as XGB
using XGBoost: DMatrix, Booster, xgboost

import SparseArrays


abstract type XGBoostAbstractRegressor <: MMI.Deterministic end
abstract type XGBoostAbstractClassifier <: MMI.Probabilistic end

const XGTypes = Union{XGBoostAbstractRegressor,XGBoostAbstractClassifier}


function _fix_objective(obj)
    if obj ∈ ("squarederror", "squaredlogerror", "logistic", "pseudohubererror", "gamma", "tweedie")
        "reg:"*obj
    elseif obj ∈ ("logistic", "logitraw", "hinge")
        "binary:"*obj
    elseif obj ∈ ("softmax", "softprob")
        "multi:"*obj
    else
        obj
    end
end

function _validate_objective(obj, startstrs)
    obj = _fix_objective(obj)
    if obj isa AbstractString
        obj == "automatic" && return true
        any(x -> startswith(obj, x), startstrs)
    elseif obj isa Union{Function,Type}
        true
    else
        false
    end
end

validate_reg_objective(obj) = _validate_objective(obj, ("reg:", "survival:"))
validate_count_objective(obj) = _validate_objective(obj, ("count:", "rank:"))
validate_class_objective(obj) = _validate_objective(obj, ("binary:", "multi:"))


function modelexpr(name::Symbol, absname::Symbol, obj::AbstractString, objvalidate::Symbol)
    quote
        MMI.@mlj_model mutable struct $name <: $absname
            test::Int = 1::(_ ≥ 0)
            num_round::Int = 100::(_ ≥ 0)  # this must be provided since XGBoost.jl doesn't provide it
            booster::String = "gbtree"
            disable_default_eval_metric::Union{Bool,Int} = 0
            eta::Float64 = 0.3::(0.0 ≤ _ ≤ 1.0)
            num_parallel_tree::Int = 1::(_ > 0)
            gamma::Float64 = 0::(_ ≥ 0)
            max_depth::Int = 6::(_ ≥ 0)
            min_child_weight::Float64 = 1::(_ ≥ 0)
            max_delta_step::Float64 = 0::(_ ≥ 0)
            subsample::Float64 = 1::(0 < _ ≤ 1)
            colsample_bytree::Float64 = 1::(0 < _ ≤ 1)
            colsample_bylevel::Float64 = 1::(0 < _ ≤ 1)
            colsample_bynode::Float64 = 1::(0 < _ ≤ 1)
            lambda::Float64 = 1::(_ ≥ 0)
            alpha::Float64 = 0::(_ ≥ 0)
            tree_method::String = "auto"
            sketch_eps::Float64 = 0.03::(0 < _ < 1)
            scale_pos_weight::Float64 = 1
            updater::Union{Nothing,String} = nothing  # this is more complicated and we don't want to set it
            refresh_leaf::Union{Int,Bool} = 1
            process_type::String = "default"
            grow_policy::String = "depthwise"
            max_leaves::Int = 0::(_ ≥ 0)
            max_bin::Int = 256::(_ ≥ 0)
            predictor::String = "cpu_predictor"
            sample_type::String = "uniform"
            normalize_type::String = "tree"
            rate_drop::Float64 = 0::(0 ≤ _ ≤ 1)
            one_drop::Union{Int,Bool} = 0::(0 ≤ _ ≤ 1)
            skip_drop::Float64 = 0::(0 ≤ _ ≤ 1)
            feature_selector::String = "cyclic"
            top_k::Int = 0::(_ ≥ 0)
            tweedie_variance_power::Float64 = 1.5::(1 < _ < 2)
            objective = $obj :: $objvalidate(_)
            base_score::Float64 = 0.5
	        early_stopping_rounds::Int = 0::(_ ≥ 0)
            watchlist = nothing  # if this is nothing we will not pass it so as to use default
            nthread::Int = Base.Threads.nthreads()::(_ ≥ 0)
            importance_type::String = "gain"
            seed::Union{Int,Nothing} = nothing  # nothing will use built in default
            # this should probably be fixed so that we don't pass invalid parameters,
            # but in the meantime, let's just disable checking
            validate_parameters::Bool = false
            eval_metric::Vector{String} = String[]
        end

    end
end


eval(modelexpr(:XGBoostRegressor, :XGBoostAbstractRegressor, "reg:squarederror", :validate_reg_objective))

function kwargs(model, verbosity, obj)
    excluded = [:importance_type]
    fn = filter(∉(excluded), fieldnames(typeof(model)))
    out = NamedTuple(n=>getfield(model, n) for n ∈ fn if !isnothing(getfield(model, n)))
    out = merge(out, (silent=(verbosity ≤ 0),))
    # watchlist is for log output, so override if it's default and verbosity ≤ 0
    wl = (verbosity ≤ 0 && isnothing(model.watchlist)) ? (;) : model.watchlist
    if !isnothing(wl)
        out = merge(out, (watchlist=wl,))
    end
    out = merge(out, (objective=_fix_objective(obj),))
    return out
end

function MMI.feature_importances(model::XGTypes, (booster, _), r)
    dict = XGB.importance(booster, model.importance_type)
    if length(last(first(dict))) > 1
        [r.features[k] => zero(first(v)) for (k, v) in dict]
    else
        [r.features[k] => first(v) for (k, v) in dict]
    end
end

function _feature_names(X, dmatrix)
    schema = Tables.schema(X)
    if schema === nothing
        [Symbol("x$j") for j in 1:ncols(dmatrix)]
    else
        schema.names |> collect
    end
end

function MMI.fit(model::XGBoostAbstractRegressor, verbosity::Integer, X, y)
    dm = DMatrix(MMI.matrix(X), float(y))
    b = xgboost(dm; kwargs(model, verbosity, model.objective)...)
    # first return value is a tuple for consistancy with classifier case
    ((b, nothing), nothing, (features=_feature_names(X, dm),))
end

MMI.predict(model::XGBoostAbstractRegressor, (booster, _), Xnew) = XGB.predict(booster, Xnew)


eval(modelexpr(:XGBoostCount, :XGBoostAbstractRegressor, "count:poisson", :validate_count_objective))


eval(modelexpr(:XGBoostClassifier, :XGBoostAbstractClassifier, "automatic", :validate_class_objective))

function MMI.fit(model::XGBoostClassifier,
                 verbosity,  # must be here even if unsupported in pkg
                 X, y,
                )
    a_target_element = y[1] # a CategoricalValue or CategoricalString
    nclass = length(MMI.classes(a_target_element))

    objective = nclass == 2 ? "binary:logistic" : "multi:softprob"
    # confusingly, xgboost only wants this set if using multi:softprob
    num_class = nclass == 2 ? (;) : (num_class=nclass,)

    # libxgboost wants float labels
    dm = DMatrix(MMI.matrix(X), float(MMI.int(y) .- 1))

    b = xgboost(dm; kwargs(model, verbosity, objective)..., num_class...)
    fr = (b, a_target_element)

    (fr, nothing, (features=_feature_names(X, dm),))
end

function MMI.predict(model::XGBoostClassifier, fitresult, Xnew)
    (result, a_target_element) = fitresult
    classes = MMI.classes(a_target_element)
    if !ismissing(result.best_iteration)
        # we can utilise the best iteration based off early stopping rounds
        o = XGB.predict(result, MMI.matrix(Xnew), ntree_limit = result.best_iteration)
    else
        o = XGB.predict(result, MMI.matrix(Xnew))
    end

    # XGB can return a rank-1 array for binary classification
    MMI.UnivariateFinite(classes, o, augment=ndims(o)==1)
end


# # SERIALIZATION


_save(fr; kw...) = XGB.save(fr, Vector{UInt8}; kw...)

_restore(fr) = XGB.load(Booster, fr)

MMI.save(::XGTypes, fr; kw...) = (_save(fr[1]; kw...), fr[2])

MMI.restore(::XGTypes, fr) = (_restore(fr[1]), fr[2])

MLJModelInterface.reports_feature_importances(::Type{<:XGBoostAbstractRegressor}) = true
MLJModelInterface.reports_feature_importances(::Type{<:XGBoostAbstractClassifier}) = true


MMI.package_name(::Type{<:XGTypes}) = "XGBoost"
MMI.package_uuid(::Type{<:XGTypes}) = "009559a3-9522-5dbb-924b-0b6ed2b22bb9"
MMI.package_url(::Type{<:XGTypes}) = "https://github.com/dmlc/XGBoost.jl"
MMI.is_pure_julia(::Type{<:XGTypes}) = false

MMI.load_path(::Type{<:XGBoostRegressor}) = "$PKG.XGBoostRegressor"
MMI.input_scitype(::Type{<:XGBoostRegressor}) = Table(Continuous)
MMI.target_scitype(::Type{<:XGBoostRegressor}) = AbstractVector{Continuous}
MMI.human_name(::Type{<:XGBoostRegressor}) = "eXtreme Gradient Boosting Regressor"

MMI.load_path(::Type{<:XGBoostCount}) = "$PKG.XGBoostCount"
MMI.input_scitype(::Type{<:XGBoostCount}) = Table(Continuous)
MMI.target_scitype(::Type{<:XGBoostCount}) = AbstractVector{Count}
MMI.human_name(::Type{<:XGBoostCount}) = "eXtreme Gradient Boosting Count Regressor"

MMI.load_path(::Type{<:XGBoostClassifier}) = "$PKG.XGBoostClassifier"
MMI.input_scitype(::Type{<:XGBoostClassifier}) = Table(Continuous)
MMI.target_scitype(::Type{<:XGBoostClassifier}) = AbstractVector{<:Finite}
MMI.human_name(::Type{<:XGBoostClassifier}) = "eXtreme Gradient Boosting Classifier"


include("docstrings.jl")


export XGBoostRegressor, XGBoostClassifier, XGBoostCount


end # module
