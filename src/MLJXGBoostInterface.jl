module MLJXGBoostInterface

import MLJModelInterface; const MMI = MLJModelInterface
using MLJModelInterface: Table, Continuous, Count, Finite, OrderedFactor, Multiclass

const PKG = "MLJXGBoostInterface"

using Tables: schema

import XGBoost as XGB
using XGBoost: DMatrix, Booster, xgboost

import SparseArrays


abstract type XGBoostAbstractRegressor <: MMI.Deterministic end
abstract type XGBoostAbstractClassifier <: MMI.Probabilistic end

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
            watchlist = nothing  # if this is nothing we will not pass it so as to use default
            nthread::Int = Base.Threads.nthreads()::(_ ≥ 0)
            importance_type::Union{Nothing,String} = "gain"
            seed::Union{Int,Nothing} = nothing  # nothing will use built in default
        end

    end
end


eval(modelexpr(:XGBoostRegressor, :XGBoostAbstractRegressor, "reg:squarederror", :validate_reg_objective))

function kwargs(model, verbosity::Integer, obj)
    excluded = [:importance_type]
    fn = filter(∉(excluded), fieldnames(typeof(model)))
    o = NamedTuple(n=>getfield(model, n) for n ∈ fn if !isnothing(getfield(model, n)))
    o = merge(o, (silent=(verbosity == 0),))
    merge(o, (objective=_fix_objective(obj),))
end

function importances(X, r)
    fs = schema(X).names
    [named_importance(fi, fs) for fi ∈ XGB.importance(r)]
end

function _getreport(b::Booster, model)
    isnothing(model.importance_type) ? (;) : (feature_importances=XGB.importance(b, model.importance_type),)
end

function MMI.fit(model::XGBoostAbstractRegressor, verbosity::Integer, X, y)
    dm = DMatrix(MMI.matrix(X), float(y))
    b = xgboost(dm; kwargs(model, verbosity, model.objective)...)
    (b, nothing, _getreport(b, model))
end

MMI.predict(model::XGBoostAbstractRegressor, fitresult, Xnew) = XGB.predict(fitresult, Xnew)


eval(modelexpr(:XGBoostCount, :XGBoostAbstractRegressor, "count:poisson", :validate_count_objective))


eval(modelexpr(:XGBoostClassifier, :XGBoostAbstractClassifier, "automatic", :validate_class_objective))

function MMI.fit(model::XGBoostClassifier
                 , verbosity::Int     #> must be here even if unsupported in pkg
                 , X
                 , y)
    a_target_element = y[1] # a CategoricalValue or CategoricalString
    nclass = length(MMI.classes(a_target_element))

    objective = nclass == 2 ? "binary:logistic" : "multi:softprob"
    # confusingly, xgboost only wants this set if using multi:softprob
    num_class = nclass == 2 ? (;) : (num_class=nclass,)

    # libxgboost wants float labels
    dm = DMatrix(MMI.matrix(X), float(MMI.int(y) .- 1))

    b = xgboost(dm; kwargs(model, verbosity, objective)..., num_class...)
    fr = (b, a_target_element)

    (fr, nothing, _getreport(b, model))
end

function MMI.predict(model::XGBoostClassifier, fitresult, Xnew)
    (result, a_target_element) = fitresult
    decode = MMI.decoder(a_target_element)
    classes = MMI.classes(a_target_element)

    o = XGB.predict(result, MMI.matrix(Xnew))

    nlevels = length(classes)
    npatterns = MMI.nrows(Xnew)

    if nlevels == 2
        true_class_probabilities = reshape(o, 1, npatterns)
        false_class_probabilities = 1 .- true_class_probabilities
        o = vcat(false_class_probabilities, true_class_probabilities)
    end

    prediction_probabilities = reshape(o, nlevels, npatterns)

    # note we use adjoint of above:
    MMI.UnivariateFinite(classes, prediction_probabilities')
end


# # SERIALIZATION


_save(fr; kw...) = XGB.save(fr, Vector{UInt8}; kw...)

_restore(fr) = XGB.load(Booster, fr)

MMI.save(::XGBoostAbstractRegressor, fr; kw...) = _save(fr; kw...)

MMI.restore(::XGBoostAbstractRegressor, fr; kw...) = _restore(fr)

MMI.save(::XGBoostClassifier, fr; kw...) = (_save(fr[1]; kw...), fr[2])

MMI.restore(::XGBoostClassifier, fr) = (_restore(fr[1]), fr[2])

MLJModelInterface.reports_feature_importances(::Type{XGBoostAbstractRegressor}) = true
MLJModelInterface.reports_feature_importances(::Type{XGBoostAbstractClassifier}) = true


const XGTypes = Union{XGBoostAbstractRegressor,XGBoostAbstractClassifier}

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
function MMI.docstring(::Type{<:XGBoostCount})
    "The XGBoost gradient boosting method, for use with `Count` univariate targets, using a Poisson objective function."
end

MMI.load_path(::Type{<:XGBoostClassifier}) = "$PKG.XGBoostClassifier"
MMI.input_scitype(::Type{<:XGBoostClassifier}) = Table(Continuous)
MMI.target_scitype(::Type{<:XGBoostClassifier}) = AbstractVector{<:Finite}
function MMI.docstring(::Type{<:XGBoostClassifier})
    "The XGBoost gradient boosting method, for use with `Finite` univariate targets \
    (`Multiclass`, `OrderedFactor` and `Binary=Finite{2}`)."
end


include("docstrings.jl")


export XGBoostRegressor, XGBoostClassifier, XGBoostCount


end # module
