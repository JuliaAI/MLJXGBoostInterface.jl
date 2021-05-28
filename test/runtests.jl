using MLJBase
using MLJSerialization
using Test
import XGBoost
using MLJXGBoostInterface
using Distributions
import StableRNGs
rng = StableRNGs.StableRNG(123)

@test_logs (:warn, r"Changing ") XGBoostClassifier(objective="wrong")
@test_logs (:warn, r"Changing ") XGBoostCount(objective="wrong")
@test_logs (:warn, r"Your") XGBoostRegressor(objective="binary:logistic")

@testset "Binary Classification" begin
    plain_classifier = XGBoostClassifier(num_round=100, seed=0)

    N=2
    X = (x1=rand(1000), x2=rand(1000), x3=rand(1000))
    ycat = map(X.x1) do x
        string(mod(round(Int, 10*x), N))
    end |> categorical

    train, test = partition(eachindex(ycat), 0.6)

    #fitresult, cache, report = MLJBase.fit(plain_classifier, 1, X, ycat;)

    m = machine(plain_classifier,X,ycat)
    fit!(m,verbosity = 0)
end

## REGRESSOR
plain_regressor = XGBoostRegressor()
n,m = 10^3, 5 ;
features = rand(rng, n,m);
weights = rand(rng, -1:1,m);
labels = features * weights;
features = MLJBase.table(features)
fitresultR, cacheR, reportR = MLJBase.fit(plain_regressor, 1, features, labels);
rpred = predict(plain_regressor, fitresultR, features);

plain_regressor.objective = "gamma"
labels = abs.(labels)
fitresultR, cacheR, reportR = MLJBase.fit(plain_regressor, 1, features, labels);
rpred = predict(plain_regressor, fitresultR, features);

importances = reportR.feature_importances

# serialization:
serializable_fitresult =
    MLJBase.save("mymodel", plain_regressor, fitresultR)

restored_fitresult = MLJBase.restore("mymodel",
                                     plain_regressor,
                                     serializable_fitresult)
@test predict(plain_regressor, restored_fitresult, features) ≈ rpred


## COUNT

count_regressor = XGBoostCount(num_round=10)

X = randn(rng, 100, 3) .* randn(rng, 3)'
Xtable = table(X)

α = 0.1
β = [-0.3, 0.2, -0.1]
λ = exp.(α .+ X * β)
ycount = [rand(rng, Poisson(λᵢ)) for λᵢ ∈ λ]

fitresultC, cacheC, reportC = MLJBase.fit(count_regressor, 1, Xtable, ycount);
cpred = predict(count_regressor, fitresultC, Xtable);

importances = reportC.feature_importances

## CLASSIFIER

plain_classifier = XGBoostClassifier(num_round=100, seed=0)

# test binary case:
N=2
X = (x1=rand(rng, 1000), x2=rand(rng, 1000), x3=rand(rng, 1000))
ycat = map(X.x1) do x
    string(mod(round(Int, 10*x), N))
end |> categorical
y = identity.(ycat) # make plain Vector with categ. elements
train, test = partition(eachindex(y), 0.6)
fitresult, cache, report = MLJBase.fit(plain_classifier, 1,
                                            selectrows(X, train), y[train];)
yhat = mode.(predict(plain_classifier, fitresult, selectrows(X, test)))
misclassification_rate = sum(yhat .!= y[test])/length(test)
@test misclassification_rate < 0.01

importances = report.feature_importances


# Multiclass{10} case:
N=10
X = (x1=rand(rng, 1000), x2=rand(rng, 1000), x3=rand(rng, 1000))
ycat = map(X.x1) do x
    string(mod(round(Int, 10*x), N))
end |> categorical
y = identity.(ycat) # make plain Vector with categ. elements

train, test = partition(eachindex(y), 0.6)
fitresult, cache, report = MLJBase.fit(plain_classifier, 1,
                                            selectrows(X, train), y[train];)
yhat = mode.(predict(plain_classifier, fitresult, selectrows(X, test)))
misclassification_rate = sum(yhat .!= y[test])/length(test)
@test misclassification_rate < 0.01

# check target pool preserved:
X = (x1=rand(rng, 400), x2=rand(rng, 400), x3=rand(rng, 400))
ycat = vcat(fill('x', 100), fill('y', 100), fill('z', 200)) |>categorical
y = identity.(ycat)
train, test = partition(eachindex(y), 0.5)
@test length(unique(y[train])) == 2
@test length(unique(y[test])) == 1
fitresult, cache, report = MLJBase.fit(plain_classifier, 1,
                                            selectrows(X, train), y[train];)
yhat = predict_mode(plain_classifier, fitresult, selectrows(X, test))
@test Set(MLJBase.classes(yhat[1])) == Set(MLJBase.classes(y[train][1]))

# serialization:
serializable_fitresult =
    MLJBase.save("mymodel", plain_classifier, fitresult)

restored_fitresult = MLJBase.restore("mymodel",
                                     plain_classifier,
                                     serializable_fitresult)

@test predict_mode(plain_classifier, restored_fitresult, selectrows(X, test)) ==
    yhat


## MACHINE INTEGRATION

# count regressor (`count_regressor`, `Xtable` and `ycount`
# defined above):

mach = machine(count_regressor, Xtable, ycount) |> fit!
yhat = predict(mach, Xtable)

# serialize:
io = IOBuffer()
MLJBase.save(io, mach)

# deserialize:
seekstart(io)
mach2 = machine(io)
close(io)

# compare:
@test predict(mach2, Xtable) ≈ yhat


# classifier
mach = machine(plain_classifier, X, y) |> fit!
yhat = predict_mode(mach, X);

# serialize:
io = IOBuffer()
MLJBase.save(io, mach)

# deserialize:
seekstart(io)
mach2 = machine(io)

# compare:
@test predict_mode(mach2, X) == yhat
