module Layers where

import Numeric.LinearAlgebra

relu :: Double -> Double
relu z
    | z < 0     = 0
    | otherwise = z

reluDerivative :: Double -> Double
reluDerivative z
    | z < 0     = 0
    | otherwise = 1

reluMatrix :: Vector Double -> Vector Double
reluMatrix = cmap relu

-- Your long list of values
inputData :: [Double]
inputData = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,38,48,48,22,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,62,97,198,243,254,254,212,27,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,67,172,254,254,225,218,218,237,248,40,0,21,164,187,0,0,0,0,0,0,0,0,0,0,0,0,0,89,219,254,97,67,14,0,0,92,231,122,23,203,236,59,0,0,0,0,0,0,0,0,0,0,0,0,25,217,242,92,4,0,0,0,0,4,147,253,240,232,92,0,0,0,0,0,0,0,0,0,0,0,0,0,101,255,92,0,0,0,0,0,0,105,254,254,177,11,0,0,0,0,0,0,0,0,0,0,0,0,0,0,167,244,41,0,0,0,7,76,199,238,239,94,10,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,121,0,0,2,63,180,254,233,126,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,190,196,14,2,97,254,252,146,52,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,130,225,71,180,232,181,60,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,130,254,254,230,46,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,6,77,244,254,162,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,110,254,218,254,116,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,131,254,154,28,213,86,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,66,209,153,19,19,233,60,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,142,254,165,0,14,216,167,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,90,254,175,0,18,229,92,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,26,229,249,176,222,244,44,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,73,193,197,134,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

inputVector :: Vector Double
inputVector = fromList inputData

type Weights = Matrix Double
type Biases  = Vector Double
type LayerOutput = Vector Double

oneHotEncoder :: Int -> Vector Double
oneHotEncoder n =
    fromList [if i == n then 1.0 else 0.0 | i <- [0..9]] 

initializeInputWeight :: Weights
initializeInputWeight = uniformSample 42 128 (replicate 784 (-0.0357, 0.0357))

initializeHiddenWeight :: Weights
initializeHiddenWeight = uniformSample 42 10 (replicate 128 (-0.088, 0.088))

initializeHiddenBias :: Biases
initializeHiddenBias = konst 0 128

initializeOutputBias :: Biases
initializeOutputBias = konst 0 10

learningRate :: Double
learningRate = 0.001

softmax :: Vector Double -> Vector Double
softmax v = let expV = exp v
                sumExpV = sumElements expV
            in expV / scalar sumExpV 


crossEntropyLoss :: Vector Double -> Vector Double -> Double
crossEntropyLoss predicted oneHot =
    - sumElements (cmap log predicted * oneHot)

shape :: (Int, Int)
shape = size initializeInputWeight

-- x is the input vector

feedForward :: (Vector Double, Weights, Weights, Biases, Biases) -> (Vector Double, Vector Double, Vector Double)
feedForward (x, w1, w2, b1, b2) = (logits, a1, z1)
    where

    z1  = (w1 #> x) + b1

    a1 = cmap relu z1

    logits  = (w2 #> a1) + b2

-- You don't actually need to pass the loss function
calculateGradients :: (Vector Double, Vector Double, Weights, Weights, Vector Double, Vector Double, Vector Double) -> (Weights, Weights, Biases, Biases)
calculateGradients (logits, oneHotEncoding, w1, w2, a1, z1, x) = (w1', w2', b1', b2')
    where
    delta2 :: Vector Double
    delta1 :: Vector Double

    delta2 = logits - oneHotEncoding
    w2' = delta2 `outer` a1
    b2' = delta2
    delta1 = (tr w2 #> delta2) * cmap reluDerivative z1
    w1' = delta1 `outer` x
    b1' = delta1

-- Stochastic Gradient Descent
updateWeights ::(Double, Weights, Weights, Biases, Biases, Weights, Weights, Biases, Biases) -> (Weights, Weights, Biases, Biases)
updateWeights (learningRate, w1, w2, b1, b2, w1', w2', b1', b2') = (neww1, neww2, newb1, newb2)
    where
    neww1 = w1 - scale learningRate w1'
    neww2 = w2 - scale learningRate w2'
    newb1 = b1 - scale learningRate b1'
    newb2 = b2 - scale learningRate b2'


-- Add learning rate
stepNN :: (Vector Double, Int, Double, Weights, Weights, Biases, Biases) -> (Weights, Weights, Biases, Biases, Double)
stepNN (x, label, learningRate, w1, w2, b1, b2) = (neww1, neww2, newb1, newb2, loss)
    where
    oneHotEncoding = oneHotEncoder label
    (logits, a1, z1) = feedForward (x, w1, w2, b1, b2)
    loss = crossEntropyLoss (softmax logits) oneHotEncoding
    (w1', w2', b1', b2') = calculateGradients (logits, oneHotEncoding, w1, w2, a1, z1, x)
    (neww1, neww2, newb1, newb2) = updateWeights (learningRate, w1, w2, b1, b2, w1', w2', b1', b2')

-- feedForward
-- crossEntropyLoss
-- Backwards
-- UpdateWeights


grain :: IO ()
grain = do

    let label = 8
    let (x, w1, w2, b1, b2) = (inputVector, initializeInputWeight, initializeHiddenWeight, initializeHiddenBias, initializeOutputBias)
    let (neww1, neww2, newb1, newb2, loss) = stepNN (x, label, learningRate, w1, w2, b1, b2)

    print loss

-- backwards pass. Hmm.
-- The gradient of the first layer is.
-- Logits - inputLabelToVector