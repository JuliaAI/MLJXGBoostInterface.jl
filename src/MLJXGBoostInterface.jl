module MLJXGBoostInterface

import MLJModelInterface; const MMI = MLJModelInterface
using MLJModelInterface: Table, Continuous, Count, Finite, OrderedFactor, Multiclass

const PKG = "MLJXGBoostInterface"

using Tables: schema

import XGBoost
using XGBoost: DMatrix, xgboost

import SparseArrays

# helper for feature importances:
# XGBoost used "f
function named_importance(fi::XGBoost.FeatureImportance, features)

    new_fname = features[parse(Int, fi.fname[2:end]) + 1] |> string
    XGBoost.FeatureImportance(new_fname, fi.gain, fi.cover, fi.freq)
end


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


function modelexpr(name::Symbol, absname::Symbol, obj::AbstractString, eval::AbstractString,
                   objvalidate::Symbol,
                  )
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
            eval_metric = $eval
            nthread::Int = Base.Threads.nthreads()::(_ ≥ 0)
            seed::Union{Int,Nothing} = nothing  # nothing will use built in default
        end

    end
end


eval(modelexpr(:XGBoostRegressor, :XGBoostAbstractRegressor, "reg:squarederror", "rmse", :validate_reg_objective))

function kwargs(model, verbosity::Integer, obj, eval=nothing)
    fn = fieldnames(typeof(model))
    o = NamedTuple(n=>getfield(model, n) for n ∈ fn if !isnothing(getfield(model, n)))
    o = merge(o, (silent=(verbosity == 0),))
    isnothing(eval) || (o = merge(o, (eval=eval,)))
    merge(o, (objective=_fix_objective(obj),))
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

    r = xgboost(dm, model.num_round; kwargs(model, verbosity, model.objective)...)

    report = (feature_importances=importances(X, r),)

    (r, nothing, report)
end

function MMI.predict(model::XGBoostAbstractRegressor, fitresult, Xnew)
    Xmatrix = _to_array(MMI.matrix(Xnew))
    XGBoost.predict(fitresult, Xmatrix)
end


eval(modelexpr(:XGBoostCount, :XGBoostAbstractRegressor, "count:poisson", "rmse", :validate_count_objective))


eval(modelexpr(:XGBoostClassifier, :XGBoostAbstractClassifier, "automatic", "mlogloss", :validate_class_objective))

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
    fr = (r, a_target_element)

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


"""
    _persistent(booster)

Private method.

Return a persistent (ie, Julia-serializable) representation of the
XGBoost.jl model `booster`.

Restore the model with [`booster`](@ref)
"""
function _persistent(booster)
    # this implemenation is not ideal; see
    # https://github.com/dmlc/XGBoost.jl/issues/103
    file = tempname()
    XGBoost.save(booster, file)
    persistent_booster = read(file)
    rm(file)
    persistent_booster
end

"""
    _booster(persistent)

Private method.

Return the XGBoost.jl model which has `persistent` as its persistent
(Julia-serializable) representation. See [`persistent`](@ref) method.
"""
function _booster(persistent)
    mktemp() do file, io
        write(io, persistent)
        close(io)
        XGBoost.Booster(model_file=file)
    end
end


MLJModelInterface.save(::XGBoostAbstractRegressor, fr; kw...) = _persistent(fr)

MLJModelInterface.restore(::XGBoostAbstractRegressor, fr) = _booster(fr)

function MLJModelInterface.save(::XGBoostClassifier, fr; kw...)
    (booster, a_target_element) = fr
    (_persistent(booster), a_target_element)
end

function MLJModelInterface.restore(::XGBoostClassifier, fr)
    (persistent, a_target_element) = fr
    (_booster(persistent), a_target_element)
end

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


include("docstrings.jl")


export XGBoostRegressor, XGBoostClassifier, XGBoostCount


end # module
