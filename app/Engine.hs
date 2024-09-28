-- Engine.hs

-- Ignore all of this. Idk if it works
-- Long story short. I took Karpathy's micrograd engine and asked GPT o1 to translate it.
-- Used up all of my prompts for the week. It kind of works?
-- Before I got to test it, I realized I could just manually calculate backpropagation.
-- So I did that instead.
-- I don't want to delete this though so here it is.

module Engine where

import qualified Data.Map.Strict as Map
import qualified Data.Set as Set
import Debug.Trace
import Numeric.LinearAlgebra

data Value = Value
  { uid :: Int
  , dataValue :: Matrix Double        -- Changed from Vector to Matrix
  , gradValue :: Matrix Double        -- Changed from Vector to Matrix
  , _backward :: Map.Map Int (Matrix Double) -> Map.Map Int (Matrix Double)
  , _prev :: [Value]
  , _op :: String
  }


instance Eq Value where
  (Value uid1 _ _ _ _ _) == (Value uid2 _ _ _ _ _) = uid1 == uid2

instance Ord Value where
  compare (Value uid1 _ _ _ _ _) (Value uid2 _ _ _ _ _) = compare uid1 uid2

-- Function to create a new Value with a unique ID
makeValue :: Int -> Matrix Double -> Value
makeValue newUid dataVal = Value
  { uid = newUid
  , dataValue = dataVal
  , gradValue = konst 0 (rows dataVal, cols dataVal)  -- Initialize gradient as a zero matrix
  , _backward = \grads -> grads                     -- Default backward function (no operation)
  , _prev = []
  , _op = ""
  }


addValue :: Int -> Value -> Value -> Value
addValue newUid a b =
  let outVal = dataValue a + dataValue b  -- Element-wise addition
      outGrad = konst 0 (rows outVal, cols outVal)  -- Initialize gradient as a zero matrix
      out = Value
              { uid = newUid
              , dataValue = outVal
              , gradValue = outGrad
              , _backward = \grads ->
                  case Map.lookup (uid out) grads of
                    Just gradOut ->
                      let grads' = Map.insertWith (+) (uid a) gradOut $
                                  Map.insertWith (+) (uid b) gradOut grads
                      in grads'
                    Nothing -> grads
              , _prev = [a, b]
              , _op = "+"
              }
  in out



mul :: Int -> Value -> Value -> Value
mul newUid a b =
  let outVal = dataValue a * dataValue b  -- Element-wise multiplication
      outGrad = konst 0 (rows outVal, cols outVal)  -- Initialize gradient as a zero matrix
      out = Value
              { uid = newUid
              , dataValue = outVal
              , gradValue = outGrad
              , _backward = \grads ->
                  case Map.lookup (uid out) grads of
                    Just gradOut ->
                      let gradA = dataValue b * gradOut  -- Element-wise multiplication
                          gradB = dataValue a * gradOut
                          grads' = Map.insertWith (+) (uid a) gradA $
                                   Map.insertWith (+) (uid b) gradB grads
                      in grads'
                    Nothing -> grads
              , _prev = [a, b]
              , _op = "*"
              }
  in out



relu :: Int -> Value -> Value
relu newUid a =
  let outVal = cmap (\x -> if x < 0 then 0 else x) (dataValue a)  -- Element-wise ReLU
      outGrad = konst 0 (rows outVal, cols outVal)  -- Initialize gradient as a zero matrix
      out = Value
              { uid = newUid
              , dataValue = outVal
              , gradValue = outGrad
              , _backward = \grads ->
                  case Map.lookup (uid out) grads of
                    Just gradOut ->
                      let mask = cmap (\x -> if x > 0 then 1 else 0) (dataValue a)  -- Derivative of ReLU
                          gradInput = mask * gradOut
                          grads' = Map.insertWith (+) (uid a) gradInput grads
                      in grads'
                    Nothing -> grads
              , _prev = [a]
              , _op = "ReLU"
              }
  in out



backward :: Value -> Map.Map Int (Matrix Double)
backward v =
  let
      topoOrder = buildTopo v Set.empty
      topoOrderRev = reverse topoOrder  -- Reverse the topological order for backpropagation
  in
      trace ("Topological order (uids): " ++ show (map uid topoOrderRev)) $
      let initialGrads = Map.singleton (uid v) (fromLists [[1.0]])  -- Initialize gradient of output as 1x1 matrix
          grads = computeGradients topoOrderRev initialGrads
      in grads
  where
    -- | Build a topological order of the computation graph
    buildTopo :: Value -> Set.Set Int -> [Value]
    buildTopo current visited =
      if uid current `Set.member` visited
      then []
      else
        let visited' = Set.insert (uid current) visited
            prevTopos = concatMap (\val -> buildTopo val visited') (_prev current)
        in prevTopos ++ [current]
    
    -- | Compute gradients by traversing the topological order
    computeGradients :: [Value] -> Map.Map Int (Matrix Double) -> Map.Map Int (Matrix Double)
    computeGradients [] grads = grads
    computeGradients (x:xs) grads =
      let grads' = _backward x grads
      in computeGradients xs grads'



sigmoid :: Int -> Value -> Value
sigmoid newUid a =
    let outVal = cmap (\x -> 1 / (1 + Prelude.exp (-x))) (dataValue a)  -- Element-wise sigmoid
        outGrad = konst 0 (rows outVal, cols outVal)  -- Initialize gradient as a zero matrix
        out = Value
                { uid = newUid
                , dataValue = outVal
                , gradValue = outGrad
                , _backward = \grads ->
                    case Map.lookup (uid out) grads of
                      Just gradOut ->
                        let sigmoidVal = outVal
                            gradInput = (sigmoidVal * (1 - sigmoidVal)) * gradOut  -- Derivative of sigmoid
                            grads' = Map.insertWith (+) (uid a) gradInput grads
                        in grads'
                      Nothing -> grads
                , _prev = [a]
                , _op = "sigmoid"
                }
    in out


-- | Matrix-Vector multiplication
matMul :: Int -> Value -> Value -> Value
matMul newUid weights a =
  let outVal = dataValue weights Numeric.LinearAlgebra.<> dataValue a  -- Matrix multiplication
      outGrad = konst 0 (rows outVal, cols outVal)  -- Initialize gradient as a zero matrix
      out = Value
              { uid = newUid
              , dataValue = outVal
              , gradValue = outGrad
              , _backward = \grads ->
                  case Map.lookup (uid out) grads of
                    Just gradOut ->
                      let gradWeights = gradOut Numeric.LinearAlgebra.<> tr (dataValue a)  -- Gradient w.r.t weights
                          gradInput = tr (dataValue weights) Numeric.LinearAlgebra.<> gradOut  -- Gradient w.r.t input a
                          grads' = Map.insertWith (+) (uid weights) gradWeights $
                                   Map.insertWith (+) (uid a) gradInput grads
                      in grads'
                    Nothing -> grads
              , _prev = [weights, a]
              , _op = "matMul"
              }
  in out

-- | Combined Softmax and Cross-Entropy Loss Function
softmaxCrossEntropyLoss :: Int -> Value -> Value -> Value
softmaxCrossEntropyLoss newUid logits oneHot =
    let
        -- Forward Pass

        -- 1. Compute max(logits) for numerical stability
        maxVal = maxElement (dataValue logits)
        shiftedData = (dataValue logits) - (konst maxVal (rows (dataValue logits), cols (dataValue logits)))

        -- 2. Exponentiate the shifted logits
        expShifted = cmap exp shiftedData

        -- 3. Compute sum of exponentials
        sumExp = sumElements expShifted

        -- 4. Compute softmax probabilities
        softmaxProb = scale (1 / sumExp) expShifted  -- p_i = exp(z_i - max(z)) / sum_j exp(z_j - max(z))

        -- 5. Compute log probabilities
        logSoftmax = cmap log softmaxProb        -- log(p_i)

        -- 6. Compute element-wise multiplication with one-hot labels
        yLogP = (dataValue oneHot) * logSoftmax  -- y_i * log(p_i)

        -- 7. Sum over all classes to get the loss
        sumYLogP = sumElements yLogP            -- sum_i y_i * log(p_i)

        -- 8. Compute final loss (negative sum)
        lossValue = - sumYLogP                  -- L = -sum_i y_i * log(p_i)

        -- 9. Create the loss Value
        loss = Value
            { uid = newUid
            , dataValue = fromLists [[lossValue]]  -- (1×1) matrix
            , gradValue = konst 0 (1,1)
            , _backward = \grads ->
                case Map.lookup newUid grads of
                    Just _ ->
                        -- Backward Pass: dL/dz = p - y
                        let gradInput = softmaxProb - (dataValue oneHot)  -- (Matrix Double)
                            grads' = Map.insertWith (+) (uid logits) gradInput grads
                        in grads'
                    Nothing -> grads
            , _prev = [logits, oneHot]
            , _op = "SoftmaxCrossEntropy"
            }
    in loss


updateGradValue :: Map.Map Int (Matrix Double) -> Value -> Value
updateGradValue gradMap v =
    v { gradValue = Map.findWithDefault (konst 0 (rows (dataValue v), cols (dataValue v))) (uid v) gradMap }


-- Function to print the Value (for debugging)
showValue :: Value -> String
showValue v = "Value(uid=" ++ show (uid v) ++ ", data=" ++ show (dataValue v) ++ ", grad=" ++ show (gradValue v) ++ ")"





{-

-- Function to compare two matrices for approximate equality
approxEqual :: Matrix Double -> Matrix Double -> Double -> Bool
approxEqual a b epsilon =
    let diff = a - b
    in maxElement (cmap abs diff) < epsilon

-- Function to run a single test case
runTest :: String -> Bool -> IO ()
runTest testName result =
    if result
        then putStrLn $ "[PASS] " ++ testName
        else putStrLn $ "[FAIL] " ++ testName

-- Top-level bindings (no need for 'let' at the top level)
weights1Mat :: Matrix Double
weights1Mat = (4 >< 4) [ 0.2, -0.3, 0.4, 0.1
                       , -0.5, 0.6, -0.7, 0.8
                       , 0.9, -1.0, 1.1, -1.2
                       , 1.3, -1.4, 1.5, -1.6 ]

bias1Vec :: Matrix Double
bias1Vec = (4 >< 1) [0.1, -0.2, 0.3, -0.4]

-- Layer 2: Hidden1 -> Hidden2
weights2Mat :: Matrix Double
weights2Mat = (4 >< 4) [ 0.1, 0.2, 0.3, 0.4
                       , 0.5, 0.6, 0.7, 0.8
                       , 0.9, 1.0, 1.1, 1.2
                       , 1.3, 1.4, 1.5, 1.6 ]

bias2Vec :: Matrix Double
bias2Vec = (4 >< 1) [0.0, 0.0, 0.0, 0.0]

-- Layer 3: Hidden2 -> Hidden3
weights3Mat :: Matrix Double
weights3Mat = (4 >< 4) [ 0.2, -0.1, 0.3, -0.4
                       , 0.5, 0.6, -0.7, 0.8
                       , 0.9, -1.0, 1.1, -1.2
                       , 1.3, -1.4, 1.5, -1.6 ]

bias3Vec :: Matrix Double
bias3Vec = (4 >< 1) [0.05, -0.05, 0.05, -0.05]

-- Layer 4: Hidden3 -> Hidden4
weights4Mat :: Matrix Double
weights4Mat = (4 >< 4) [ 0.1, 0.2, 0.3, 0.4
                       , 0.5, 0.6, 0.7, 0.8
                       , 0.9, 1.0, 1.1, 1.2
                       , 1.3, 1.4, 1.5, 1.6 ]

bias4Vec :: Matrix Double
bias4Vec = (4 >< 1) [0.0, 0.0, 0.0, 0.0]

-- Output Layer: Hidden4 -> Output
weights5Mat :: Matrix Double
weights5Mat = (4 >< 4) [ 0.2, -0.3, 0.4, -0.5
                       , 0.6, -0.7, 0.8, -0.9
                       , 1.0, -1.1, 1.2, -1.3
                       , 1.4, -1.5, 1.6, -1.7 ]

bias5Vec :: Matrix Double
bias5Vec = (4 >< 1) [0.1, -0.1, 0.1, -0.1]

-- Creating Value instances for weights and biases
weights1, bias1, weights2, bias2, weights3, bias3, weights4, bias4, weights5, bias5 :: Value
weights1 = makeValue 4 weights1Mat
bias1    = makeValue 5 bias1Vec
weights2 = makeValue 6 weights2Mat
bias2    = makeValue 7 bias2Vec
weights3 = makeValue 8 weights3Mat
bias3    = makeValue 9 bias3Vec
weights4 = makeValue 10 weights4Mat
bias4    = makeValue 11 bias4Vec
weights5 = makeValue 12 weights5Mat
bias5    = makeValue 13 bias5Vec

-- Example input as a 4×1 matrix
inputMatrix :: Matrix Double
inputMatrix = (4 >< 1) [1.0, 2.0, 3.0, 4.0]

x :: Value
x = makeValue 0 inputMatrix

main :: IO ()
main = do
    putStrLn "Running Unit Tests...\n"
    
    -- **Unit Test 1: addValue**
    let
        -- Define two simple 2x2 matrices
        a1 = makeValue 100 (fromLists [[1.0, 2.0], [3.0, 4.0]])
        a2 = makeValue 101 (fromLists [[5.0, 6.0], [7.0, 8.0]])
        
        -- Perform addition
        addTest = addValue 102 a1 a2
        
        -- Define a simple loss: sum of all elements in addTest
        lossAdd = sumValue 103 addTest
        
        -- Perform backward pass
        gradMapAdd = backward lossAdd
        
        -- Update gradients
        updatedAddTest = updateGradValue gradMapAdd addTest
        updatedA1 = updateGradValue gradMapAdd a1
        updatedA2 = updateGradValue gradMapAdd a2
        
        -- Expected gradients: dL/d(add) = 1, so dL/da1 = 1, dL/da2 = 1
        expectedGradAdd = konst 1 (rows (dataValue addTest), cols (dataValue addTest))
        
        -- Check gradients
        gradAddCorrect = approxEqual (gradValue updatedAddTest) expectedGradAdd 1e-6
        gradA1Correct = approxEqual (gradValue updatedA1) expectedGradAdd 1e-6
        gradA2Correct = approxEqual (gradValue updatedA2) expectedGradAdd 1e-6
    
    runTest "addValue - Gradient w.r.t. addTest" gradAddCorrect
    runTest "addValue - Gradient w.r.t. a1" gradA1Correct
    runTest "addValue - Gradient w.r.t. a2" gradA2Correct
    
    -- **Unit Test 2: mul**
    let
        -- Define two simple 2x2 matrices
        b1 = makeValue 110 (fromLists [[2.0, 3.0], [4.0, 5.0]])
        b2 = makeValue 111 (fromLists [[6.0, 7.0], [8.0, 9.0]])
        
        -- Perform element-wise multiplication
        mulTest = mul 112 b1 b2
        
        -- Define a simple loss: sum of all elements in mulTest
        lossMul = sumValue 113 mulTest
        
        -- Perform backward pass
        gradMapMul = backward lossMul
        
        -- Update gradients
        -- Removed 'updatedMulTest' as it's unused
        updatedB1 = updateGradValue gradMapMul b1
        updatedB2 = updateGradValue gradMapMul b2
        
        -- Expected gradients:
        -- dL/d(mul) = 1
        -- dL/db1 = b2
        -- dL/db2 = b1
        expectedGradB1 = dataValue b2
        expectedGradB2 = dataValue b1
    
        -- Check gradients
        gradB1Correct = approxEqual (gradValue updatedB1) expectedGradB1 1e-6
        gradB2Correct = approxEqual (gradValue updatedB2) expectedGradB2 1e-6
    
    runTest "mul - Gradient w.r.t. b1" gradB1Correct
    runTest "mul - Gradient w.r.t. b2" gradB2Correct
      
    -- **Unit Test 3: relu**
    let
        -- Define a simple 2x2 matrix
        c1 = makeValue 120 (fromLists [[-1.0, 2.0], [3.0, -4.0]])
        
        -- Apply ReLU
        reluTest = relu 121 c1
        
        -- Define a simple loss: sum of all elements in reluTest
        lossRelu = sumValue 122 reluTest
        
        -- Perform backward pass
        gradMapRelu = backward lossRelu
        
        -- Update gradients
        -- Removed 'updatedReluTest' as it's unused
        updatedC1 = updateGradValue gradMapRelu c1
        
        -- Expected gradients:
        -- dL/d(relu) = 1
        -- dL/d(c1) = mask where c1 > 0
        mask = cmap (\value -> if value > 0 then 1 else 0) (dataValue c1)
        expectedGradC1 = mask
        
        -- Check gradients
        gradC1Correct = approxEqual (gradValue updatedC1) expectedGradC1 1e-6
    
    runTest "relu - Gradient w.r.t. c1" gradC1Correct
    
    -- **Unit Test 4: matMul**
    let
        -- Define weights (2x2) and input (2x1)
        dWeights = makeValue 130 (fromLists [[1.0, 2.0], [3.0, 4.0]])
        dInput = makeValue 131 (fromLists [[5.0], [6.0]])
        
        -- Perform matrix multiplication
        matMulTest = matMul 132 dWeights dInput  -- (2x2) * (2x1) = (2x1)
        
        -- Define a simple loss: sum of all elements in matMulTest
        lossMatMul = sumValue 133 matMulTest
        
        -- Perform backward pass
        gradMapMatMul = backward lossMatMul
        
        -- Update gradients
        -- Removed 'updatedMatMulTest' as it's unused
        updatedWeights = updateGradValue gradMapMatMul dWeights
        updatedInput = updateGradValue gradMapMatMul dInput
        
        -- Expected gradients:
        -- dL/d(matMul) = 1
        -- dL/d(weights) = gradOut * input^T
        -- dL/d(input) = weights^T * gradOut
        gradOutMatMul = konst 1 (rows (dataValue matMulTest), cols (dataValue matMulTest))
        expectedGradWeights = gradOutMatMul Numeric.LinearAlgebra.<> tr (dataValue dInput)  -- (2x1) * (1x2) = (2x2)
        expectedGradInput = tr (dataValue dWeights) Numeric.LinearAlgebra.<> gradOutMatMul  -- (2x2)^T * (2x1) = (2x1)
    
        -- Check gradients
        gradWeightsCorrect = approxEqual (gradValue updatedWeights) expectedGradWeights 1e-6
        gradInputCorrect = approxEqual (gradValue updatedInput) expectedGradInput 1e-6
    
    runTest "matMul - Gradient w.r.t. weights" gradWeightsCorrect
    runTest "matMul - Gradient w.r.t. input" gradInputCorrect
    
    -- **Unit Test 5: logValue**
    let
        -- Define a simple 2x1 matrix with positive values
        e1 = makeValue 140 (fromLists [[2.0], [3.0]])
        
        -- Apply log
        logTest = logValue 141 e1
        
        -- Define a simple loss: sum of all elements in logTest
        lossLog = sumValue 142 logTest
        
        -- Perform backward pass
        gradMapLog = backward lossLog
        
        -- Update gradients
        -- Removed 'updatedLogTest' as it's unused
        updatedE1 = updateGradValue gradMapLog e1
        
        -- Expected gradients:
        -- dL/d(log) = 1
        -- dL/de1 = 1 / e1
        expectedGradE1 = cmap (\x -> 1 / x) (dataValue e1)
    
        -- Check gradients
        gradE1Correct = approxEqual (gradValue updatedE1) expectedGradE1 1e-6
    
    runTest "logValue - Gradient w.r.t. e1" gradE1Correct
    
    -- **Unit Test 6: sumValue**
    let
        -- Define a simple 2x2 matrix
        f1 = makeValue 150 (fromLists [[1.0, 2.0], [3.0, 4.0]])
        
        -- Perform sum
        sumTest = sumValue 151 f1
        
        -- Define a simple loss: the sum itself (identity)
        lossSum = sumTest  -- Already the sum
        
        -- Perform backward pass
        gradMapSum = backward lossSum
        
        -- Update gradients
        -- Removed 'updatedSumTest' as it's unused
        updatedF1 = updateGradValue gradMapSum f1
        
        -- Expected gradients:
        -- dL/d(sum) = 1
        -- dL/df1 = 1 (since sum over f1)
        expectedGradF1 = konst 1 (rows (dataValue f1), cols (dataValue f1))
    
        -- Check gradients
        gradF1Correct = approxEqual (gradValue updatedF1) expectedGradF1 1e-6
    
    runTest "sumValue - Gradient w.r.t. f1" gradF1Correct
    
    putStrLn "\nUnit Tests Completed.\n"
    
    -- **Original Forward and Backward Pass**
    putStrLn "Running Forward and Backward Pass...\n"
    
    -- Forward pass through the network
    let
        -- Input layer
        input = x

        -- Layer 1: Input -> Hidden1
        mat1 = matMul 14 weights1 input
        add1 = addValue 15 mat1 bias1
        relu1 = relu 16 add1

        -- Layer 2: Hidden1 -> Hidden2
        mat2 = matMul 17 weights2 relu1
        add2 = addValue 18 mat2 bias2
        relu2 = relu 19 add2

        -- Layer 3: Hidden2 -> Hidden3
        mat3 = matMul 20 weights3 relu2
        add3 = addValue 21 mat3 bias3
        relu3 = relu 22 add3

        -- Layer 4: Hidden3 -> Hidden4
        mat4 = matMul 23 weights4 relu3
        add4 = addValue 24 mat4 bias4
        relu4 = relu 25 add4

        -- Output Layer: Hidden4 -> Output
        mat5 = matMul 26 weights5 relu4
        add5 = addValue 27 mat5 bias5
        output = sigmoid 28 add5

        -- One-hot encoded label (adjust the length based on your number of classes)
        oneHotVector = makeValue 29 (fromLists [[0.0], [0.0], [1.0], [0.0]])  -- 4×1 matrix

        -- Compute cross-entropy loss
        loss = crossEntropyLossFunc 30 output oneHotVector

    -- Print the Values before backward pass
    putStrLn "Before backward pass:"
    mapM_ (putStrLn . showValue) [input, mat1, add1, relu1, mat2, add2, relu2, mat3, add3, relu3, mat4, add4, relu4, mat5, add5, output, oneHotVector, loss]

    -- Perform backward pass to compute the gradients
    let gradMap = backward loss

    -- Print the gradient map (for debugging)
    putStrLn "\nGradient map:"
    mapM_ (putStrLn . show) (Map.toList gradMap)

    -- Update the Values with their computed gradients
    let allValues = [x, weights1, bias1, mat1, add1, relu1,
                    weights2, bias2, mat2, add2, relu2,
                    weights3, bias3, mat3, add3, relu3,
                    weights4, bias4, mat4, add4, relu4,
                    weights5, bias5, mat5, add5, output, oneHotVector, loss]
        updatedValues = map (updateGradValue gradMap) allValues

    -- Print the Values after backward pass
    putStrLn "\nAfter backward pass:"
    mapM_ (putStrLn . showValue) updatedValues

    -- **Print Final Loss Value**
    let finalLoss = dataValue loss
    putStrLn "\nFinal Loss:"
    print finalLoss

-}