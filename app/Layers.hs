import Numeric.LinearAlgebra



class Differentiable f where
    pD :: f -> Double -> Double

relu :: Double -> Double
relu z
    | z < 0     = 0
    | otherwise = z

reluMatrix :: Vector Double -> Vector Double
reluMatrix = cmap relu

-- Your long list of values
inputData :: [Double]
inputData = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,38,48,48,22,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,62,97,198,243,254,254,212,27,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,67,172,254,254,225,218,218,237,248,40,0,21,164,187,0,0,0,0,0,0,0,0,0,0,0,0,0,89,219,254,97,67,14,0,0,92,231,122,23,203,236,59,0,0,0,0,0,0,0,0,0,0,0,0,25,217,242,92,4,0,0,0,0,4,147,253,240,232,92,0,0,0,0,0,0,0,0,0,0,0,0,0,101,255,92,0,0,0,0,0,0,105,254,254,177,11,0,0,0,0,0,0,0,0,0,0,0,0,0,0,167,244,41,0,0,0,7,76,199,238,239,94,10,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,121,0,0,2,63,180,254,233,126,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,190,196,14,2,97,254,252,146,52,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,130,225,71,180,232,181,60,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,130,254,254,230,46,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,6,77,244,254,162,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,110,254,218,254,116,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,131,254,154,28,213,86,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,66,209,153,19,19,233,60,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,142,254,165,0,14,216,167,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,90,254,175,0,18,229,92,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,26,229,249,176,222,244,44,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,73,193,197,134,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

inputVector :: Vector Double
inputVector = fromList inputData

newtype InputLabel = InputLabel Int
inputLabel = 8

inputLabelToVector :: InputLabel -> Vector Double
inputLabelToVector (InputLabel n) =
    fromList [if i == n then 1.0 else 0.0 | i <- [0..9]] 

initializeInputWeight :: Matrix Double
initializeInputWeight = uniformSample 42 128 (replicate 784 (-0.0357, 0.0357))

initializeHiddenWeight :: Matrix Double
initializeHiddenWeight = uniformSample 42 10 (replicate 128 (-0.088, 0.088))

initializeHiddenBias :: Vector Double
initializeHiddenBias = konst 0 128

initializeOutputBias :: Vector Double
initializeOutputBias = konst 0 10

sigmoid :: Double -> Double
sigmoid x = 1 / (1 + exp (-x))

softmax :: Vector Double -> Vector Double
softmax v = let expV = exp v
                sumExpV = sumElements expV
            in expV / scalar sumExpV 

shape :: (Int, Int)
shape = size initializeInputWeight

feedForward :: Vector Double -> IO (Vector Double)
feedForward input = do
    -- Compute hidden layer input
    let hiddenLayerInput  = (initializeInputWeight #> input) + initializeHiddenBias
    putStrLn "\nHidden Layer Input:"
    print hiddenLayerInput

    -- Apply ReLU to hidden layer output
    let hiddenLayerOutput = cmap relu hiddenLayerInput
    putStrLn "\nHidden Layer Output (After ReLU):"
    print hiddenLayerOutput

    -- Compute output layer input
    let outputLayerInput  = (initializeHiddenWeight #> hiddenLayerOutput) + initializeOutputBias
    putStrLn "\nOutput Layer Input (Final Output):"
    print outputLayerInput

    let output = softmax outputLayerInput
    print output

    return output

crossEntropyLoss :: Vector Double -> Vector Double -> Double
crossEntropyLoss predicted oneHot =
    - sumElements (cmap log predicted * oneHot)

main :: IO ()
main = do
    -- Perform a feedforward pass and print each step
    output <- feedForward inputVector
    putStrLn "\nFinal Output:"
    print output

    let label = InputLabel 3  -- For example, the label is 3
    let oneHotVector = inputLabelToVector label
    putStrLn "One-hot encoded vector:"
    print oneHotVector

    let dotProduct = output <.> oneHotVector
    print dotProduct

    print (cmap log output)

    let loss = crossEntropyLoss output oneHotVector
    putStrLn "\nCross Entropy Loss:"
    print loss
