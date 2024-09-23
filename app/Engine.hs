import qualified Data.Map.Strict as Map
import qualified Data.Set as Set
import Debug.Trace
import Numeric.LinearAlgebra

data Value = Value
  { uid :: Int                 -- Unique identifier
  , dataValue :: Double        -- The scalar value
  , gradValue :: Double        -- The gradient of the value
  , _backward :: Map.Map Int Double -> Map.Map Int Double -- Function for backpropagation
  , _prev :: [Value]           -- The previous nodes in the computation graph
  , _op :: String              -- The operation that produced this value
  }

instance Eq Value where
  (Value uid1 _ _ _ _ _) == (Value uid2 _ _ _ _ _) = uid1 == uid2

instance Ord Value where
  compare (Value uid1 _ _ _ _ _) (Value uid2 _ _ _ _ _) = compare uid1 uid2

-- Function to create a new Value with a unique ID
makeValue :: Int -> Double -> Value
makeValue newUid dataVal = Value
  { uid = newUid
  , dataValue = dataVal
  , gradValue = 0.0
  , _backward = \grads -> grads
  , _prev = []
  , _op = ""
  }

add :: Int -> Value -> Value -> Value
add newUid a b =
  let out = Value
              { uid = newUid
              , dataValue = dataValue a + dataValue b
              , gradValue = 0
              , _backward = \grads ->
                  let gradOut = Map.findWithDefault 0 (uid out) grads
                      grads' = Map.insertWith (+) (uid a) gradOut grads
                      grads'' = Map.insertWith (+) (uid b) gradOut grads'
                  in grads''
              , _prev = [a, b]
              , _op = "+"
              }
  in out

mul :: Int -> Value -> Value -> Value
mul newUid a b =
  let out = Value
              { uid = newUid
              , dataValue = dataValue a * dataValue b
              , gradValue = 0
              , _backward = \grads ->
                  let gradOut = Map.findWithDefault 0 (uid out) grads
                      grads' = Map.insertWith (+) (uid a) (dataValue b * gradOut) grads
                      grads'' = Map.insertWith (+) (uid b) (dataValue a * gradOut) grads'
                  in grads''
              , _prev = [a, b]
              , _op = "*"
              }
  in out

relu :: Int -> Value -> Value
relu newUid a =
  let outVal = if dataValue a < 0 then 0 else dataValue a
      out = Value
              { uid = newUid
              , dataValue = outVal
              , gradValue = 0
              , _backward = \grads ->
                  let gradOut = Map.findWithDefault 0 (uid out) grads
                      gradInput = if dataValue out > 0 then gradOut else 0
                      grads' = Map.insertWith (+) (uid a) gradInput grads
                  in grads'
              , _prev = [a]
              , _op = "ReLU"
              }
  in out

backward :: Value -> Map.Map Int Double
backward v =
  let
      topoOrder = buildTopo v Set.empty
      topoOrderRev = reverse topoOrder  -- Reverse the topological order
  in
      trace ("Topological order (uids): " ++ show (map uid topoOrderRev)) $
      let initialGrads = Map.singleton (uid v) 1.0
          grads = computeGradients topoOrderRev initialGrads
      in grads
  where
    buildTopo :: Value -> Set.Set Int -> [Value]
    buildTopo current visited =
      if uid current `Set.member` visited
      then []
      else
        let visited' = Set.insert (uid current) visited
            prevTopos = concatMap (\val -> buildTopo val visited') (_prev current)
        in prevTopos ++ [current]
    
    computeGradients :: [Value] -> Map.Map Int Double -> Map.Map Int Double
    computeGradients [] grads = grads
    computeGradients (x:xs) grads =
      let grads' = _backward x grads
      in computeGradients xs grads'

sigmoid :: Int -> Value -> Value
sigmoid newUid a =
    let outVal = 1 / (1 + Prelude.exp (-dataValue a))
        out = Value
                { uid = newUid
                , dataValue = outVal
                , gradValue = 0
                , _backward = \grads ->
                    let gradOut = Map.findWithDefault 0 (uid out) grads
                        sigmoidVal = dataValue out
                        gradInput = sigmoidVal * (1 - sigmoidVal) * gradOut
                        grads' = Map.insertWith (+) (uid a) gradInput grads
                    in grads'
                , _prev = [a]
                , _op = "sigmoid"
                }
    in out


-- Function to update the gradValue of each Value using the gradient map
updateGradValue :: Map.Map Int Double -> Value -> Value
updateGradValue gradMap v =
    v { gradValue = Map.findWithDefault 0 (uid v) gradMap }

-- Function to print the Value (for debugging)
showValue :: Value -> String
showValue v = "Value(uid=" ++ show (uid v) ++ ", data=" ++ show (dataValue v) ++ ", grad=" ++ show (gradValue v) ++ ")"

main :: IO ()
main = do
    -- Create the input value x
    let x = makeValue 0 0.0  -- x = 0.0

        -- Perform the operations: f = sigmoid(x)
        f = sigmoid 1 x  -- f = sigmoid(x)

    -- Print the values before backward
    putStrLn "Before backward pass:"
    mapM_ (putStrLn . showValue) [x, f]

    -- Perform the backward pass to compute the gradients
    let gradMap = backward f

    -- Print the gradient map (for debugging)
    putStrLn $ "\nGradient map: " ++ show gradMap

    -- Update the Values with their computed gradients
    let updatedValues = map (updateGradValue gradMap) [x, f]
        [x', f'] = updatedValues

    -- Print the values after backward
    putStrLn "\nAfter backward pass:"
    mapM_ (putStrLn . showValue) [x', f']
