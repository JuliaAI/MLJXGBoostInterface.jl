using MLJBase
using Test
import XGBoost
using MLJXGBoostInterface
using MLJTestInterface
using Distributions
using StableRNGs
const rng = StableRNG(123)

@test_logs (:warn, r"Constraint ") XGBoostClassifier(objective="wrong")
@test_logs (:warn, r"Constraint ") XGBoostCount(objective="wrong")
@test_logs (:warn, r"Constraint ") XGBoostRegressor(objective="binary:logistic")

@testset "Binary Classification" begin
    plain_classifier = XGBoostClassifier(num_round=100, seed=0)

    N=2
    X = (x1=rand(1000), x2=rand(1000), x3=rand(1000))
    ycat = map(X.x1) do x
        string(mod(round(Int, 10*x), N))
    end |> categorical

    train, test = partition(eachindex(ycat), 0.6)

    #fitresult, cache, report = MLJBase.fit(plain_classifier, 1, X, ycat;)

    m = machine(plain_classifier, X, ycat)
    fit!(m,verbosity = 0)
end

@testset "regressor" begin
    plain_regressor = XGBoostRegressor()
    (n, m) = (10^3, 5)
    features = rand(rng, n,m);
    weights = rand(rng, -1:1,m);
    labels = features * weights;
    features = MLJBase.table(features)
    (fitresultR, cacheR, reportR) = MLJBase.fit(plain_regressor, 0, features, labels)
    rpred = predict(plain_regressor, fitresultR, features);

    plain_regressor.objective = "gamma"
    labels = abs.(labels)
    fitresultR, cacheR, reportR = MLJBase.fit(plain_regressor, 0, features, labels)
    rpred = predict(plain_regressor, fitresultR, features);

    # serialization:
    serializable_fitresult = MLJBase.save(plain_regressor, fitresultR)

    restored_fitresult = MLJBase.restore(plain_regressor, serializable_fitresult)
    @test predict(plain_regressor, restored_fitresult, features) ≈ rpred

    imps = feature_importances(plain_regressor, fitresultR, reportR)
    @test Set(string.([imp[1] for imp ∈ imps])) == Set(string.(("x",), 1:5))

    # test regressor for early stopping rounds
    # add some noise to create more differentiator in the evaluation metric to test if it chose the correct ntree_limit
    mod_labels = labels + rand(StableRNG(123), Float64, 1000) * 10
    es_regressor = XGBoostRegressor(num_round = 250, early_stopping_rounds = 20, eta = 0.5, max_depth = 20, 
        eval_metric = ["mae"], watchlist = Dict("train" => XGBoost.DMatrix(features, mod_labels)))
    (fitresultR, cacheR, reportR) = @test_logs(
        (:info,),
        match_mode=:any,
        MLJBase.fit(es_regressor, 0, features, mod_labels),
    )
    rpred = predict(es_regressor, fitresultR, features);
    @test abs(mean(abs.(rpred-mod_labels)) - fitresultR[1].best_score) < 1e-8
    @test !ismissing(fitresultR[1].best_iteration)
    
    # try without early stopping (should be worse given the generated dataset) - to make sure it's a fair comparison - set early_stopping_rounds = num_round
    nes_regressor = XGBoostRegressor(num_round = 250, early_stopping_rounds = 250, eta = 0.5, max_depth = 20, 
        eval_metric = ["mae"], watchlist = Dict("train" => XGBoost.DMatrix(features, mod_labels)))
    (fitresultR, cacheR, reportR) = @test_logs(
        (:info,),
        match_mode=:any,
        MLJBase.fit(nes_regressor, 0, features, mod_labels),
    )
    rpred_noES = predict(es_regressor, fitresultR, features);
    @test abs(mean(abs.(rpred-mod_labels))) < abs(mean(abs.(rpred_noES-mod_labels)))
    @test ismissing(fitresultR[1].best_iteration)
end

@testset "count" begin
    count_regressor = XGBoostCount(num_round=10)

    X = randn(rng, 100, 3) .* randn(rng, 3)'
    Xtable = table(X)

    α = 0.1
    β = [-0.3, 0.2, -0.1]
    λ = exp.(α .+ X * β)
    ycount_ = [rand(rng, Poisson(λᵢ)) for λᵢ ∈ λ]
    ycount = @view(ycount_[:]) # intention is to simulate issue #17

    fitresultC, cacheC, reportC = MLJBase.fit(count_regressor, 0, Xtable, ycount);
    fitresultC_, cacheC_, reportC_ = MLJBase.fit(count_regressor, 0, Xtable, ycount_);
    # the `cacheC` and `reportC` should be same for both models but the
    # `fitresultC`s might be different since they may have different pointers to same
    # information.
    @test cacheC == cacheC_
    @test reportC == reportC_
    cpred = predict(count_regressor, fitresultC, Xtable);

    imps = feature_importances(count_regressor, fitresultC, reportC)
    @test Set(string.([imp[1] for imp ∈ imps])) == Set(string.(("x",), 1:3))

    ser = MLJBase.save(count_regressor, fitresultC)
    restored_fitresult = MLJBase.restore(count_regressor, ser)
    @test predict(count_regressor, restored_fitresult, Xtable) ≈ cpred
end

@testset "classifier" begin
    plain_classifier = XGBoostClassifier(num_round=100, seed=0)

    # test binary case:
    N=2
    X = (x1=rand(rng, 1000), x2=rand(rng, 1000), x3=rand(rng, 1000))
    ycat = map(X.x1) do x
        string(mod(round(Int, 10*x), N))
    end |> categorical
    y = identity.(ycat) # make plain Vector with categ. elements
    train, test = partition(eachindex(y), 0.6)
    fitresult, cache, report = MLJBase.fit(plain_classifier, 0,
                                                selectrows(X, train), y[train];)
    yhat = mode.(predict(plain_classifier, fitresult, selectrows(X, test)))
    misclassification_rate = sum(yhat .!= y[test])/length(test)
    @test misclassification_rate < 0.025

    # Multiclass{10} case:
    N=10
    X = (x1=rand(rng, 1000), x2=rand(rng, 1000), x3=rand(rng, 1000))
    ycat = map(X.x1) do x
        string(mod(round(Int, 10*x), N))
    end |> categorical
    y = identity.(ycat) # make plain Vector with categ. elements

    train, test = partition(eachindex(y), 0.6)
    fitresult, cache, report = MLJBase.fit(plain_classifier, 0,
                                                selectrows(X, train), y[train];)
    fitresult_, cache_, report_ = MLJBase.fit(
        plain_classifier, 0, selectrows(X, train), @view(y[train]);
    ) # mimick issue #17
    # the `cache` and `report` should be same for both models but the
    # `fitresult` might be different since they may have different pointers to same
    # information.
    @test cache == cache_
    @test report == report_

    yhat = mode.(predict(plain_classifier, fitresult, selectrows(X, test)))
    misclassification_rate = sum(yhat .!= y[test])/length(test)
    @test misclassification_rate < 0.03

    # check target pool preserved:
    X = (x1=rand(rng, 400), x2=rand(rng, 400), x3=rand(rng, 400))
    ycat = vcat(fill('x', 100), fill('y', 100), fill('z', 200)) |>categorical
    y = identity.(ycat)
    train, test = partition(eachindex(y), 0.5)
    @test length(unique(y[train])) == 2
    @test length(unique(y[test])) == 1
    fitresult, cache, report = MLJBase.fit(plain_classifier, 0,
                                                selectrows(X, train), y[train];)
    yhat = predict_mode(plain_classifier, fitresult, selectrows(X, test))
    @test Set(MLJBase.classes(yhat[1])) == Set(MLJBase.classes(y[train][1]))

    # serialization:
    serializable_fitresult = MLJBase.save(plain_classifier, fitresult)

    restored_fitresult = MLJBase.restore(plain_classifier, serializable_fitresult)

    @test predict_mode(plain_classifier, restored_fitresult, selectrows(X, test)) == yhat
end

# we repeat some of the above for sake of keeping testsets separate
# could use some generation functions...
@testset "machine" begin
    count_regressor = XGBoostCount(num_round=10)

    plain_classifier = XGBoostClassifier(num_round=100, seed=0)

    X = randn(rng, 100, 3) .* randn(rng, 3)'
    Xtable = table(X)

    α = 0.1
    β = [-0.3, 0.2, -0.1]
    λ = exp.(α .+ X * β)
    ycount_ = [rand(rng, Poisson(λᵢ)) for λᵢ ∈ λ]
    ycount = @view(ycount_[:]) # intention is to simulate issue #17

    mach = machine(count_regressor, Xtable, ycount)
    fit!(mach, verbosity=0)
    yhat = predict(mach, Xtable)

    weight = rand(length(ycount))
    mach_withweight = machine(count_regressor, Xtable, ycount, weight)
    fit!(mach_withweight, verbosity=0)
    yhat_withweight = predict(mach_withweight, Xtable)

    @test yhat !≈ yhat_withweight

    # serialize:
    io = IOBuffer()
    MLJBase.save(io, mach)

    # deserialize:
    seekstart(io)
    mach2 = machine(io)
    close(io)

    # compare:
    @test predict(mach2, Xtable) ≈ yhat

    N=10
    X = (x1=rand(rng, 1000), x2=rand(rng, 1000), x3=rand(rng, 1000))
    ycat = map(X.x1) do x
        string(mod(round(Int, 10*x), N))
    end |> categorical
    yclass = identity.(ycat) # make plain Vector with categ. elements

    # classifier
    mach = machine(plain_classifier, X, yclass)
    fit!(mach, verbosity=0)
    yhat = predict_mode(mach, X);

    imps = feature_importances(mach)
    @test Set(string.([imp[1] for imp ∈ imps])) == Set(string.(("x",), 1:3))

    # serialize:
    io = IOBuffer()
    MLJBase.save(io, mach)

    # deserialize:
    seekstart(io)
    mach2 = machine(io)

    # compare:
    @test predict_mode(mach2, X) == yhat
end

@testset "generic interface tests" begin
    @testset "Default Early Stopping Params" begin
        @test XGBoostRegressor().early_stopping_rounds == 0
    end
    @testset "XGBoostRegressor" begin
        failures, summary = MLJTestInterface.test(
            [XGBoostRegressor,],
            MLJTestInterface.make_regression()...;
            mod=@__MODULE__,
            verbosity=0, # bump to debug
            throw=false, # set to true to debug
        )
        @test isempty(failures)
    end
    @testset "XGBoostCount" begin
        failures, summary = MLJTestInterface.test(
            [XGBoostCount],
            MLJTestInterface.make_count()...;
            mod=@__MODULE__,
            verbosity=0, # bump to debug
            throw=false, # set to true to debug
        )
        @test isempty(failures)
    end
    @testset "XGBoostClassifier" begin
        for data in [
            MLJTestInterface.make_binary(),
            MLJTestInterface.make_multiclass(),
        ]
            failures, summary = MLJTestInterface.test(
                [XGBoostClassifier],
                data...;
                mod=@__MODULE__,
                verbosity=0, # bump to debug
                throw=false, # set to true to debug
            )
            @test isempty(failures)
        end
    end
end
