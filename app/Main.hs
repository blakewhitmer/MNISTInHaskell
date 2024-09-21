{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances #-}

module Main where

import qualified Data.ByteString.Lazy as ByteString
import System.Random
import Data.Array

main :: IO ()
main = do
    let file = "/Users/blakewhitmer/Downloads/archive/train-images.idx3-ubyte"
    content <- ByteString.readFile file
    putStrLn $ "File size: " ++ show (ByteString.length content) ++ " bytes"
    -- let strippedcontent = ByteString.drop 16 content
    -- let firstsixteen = ByteString.take 16 content
    -- putStrLn $ "File size: " ++ show (ByteString.length strippedcontent) ++ " bytes"
    -- putStrLn $ "First 16 (bytes): " ++ show (ByteString.unpack firstsixteen)

    -- printList strippedcontent

    randomArray <- generateRandomArray
    -- Print the array in a more readable format
    mapM_ (putStrLn . show) (elems randomArray)  

printList :: ByteString.ByteString -> IO ()
printList bs
    | ByteString.null bs = return ()  -- Base case: stop when ByteString is empty
    | otherwise = do
        let firsttwentyeight = ByteString.take 28 bs
        putStrLn (show (ByteString.unpack firsttwentyeight))  -- Convert to list of bytes for printing
        let restoflist = ByteString.drop 28 bs
        printList restoflist

generateRandomArray :: IO (Array Int Int)
generateRandomArray = do
    -- Generate a list of 256 random integers
    randomValues <- sequence $ replicate 256 (getStdRandom (randomR (0, 100)))
    -- Return the array with index range (1, 256) and corresponding random values
    return $ array (1, 256) (zip [1..256] randomValues)

class Differentiable f where
    pD :: f -> Float -> Float

newtype ReLU = ReLU (Float -> Float)

relu :: Float -> Float
relu z
    | z < 0     = 0
    | otherwise = z
instance Differentiable ReLU where
    pD (ReLU f) z
        | z < 0     = 0  -- Derivative of ReLU when z < 0
        | otherwise = 1  -- Derivative of ReLU when z >= 0

class Differentiable f where
    pD :: f -> Array Int Float -> Int -> Float

newtype Weight = Weight (Array Int Float)

newtype WeightFunction (Weight -> Weight -> Weight)





weight :: Float -> Float -> Float
weight w x = w * x
instance Differentiable Weight where
    pD (Weight f) w = f 1 w

newtype Bias = Bias (Float -> Float -> Float)

-- Define the bias function (bias is constant, so it ignores x)
bias :: Float -> Float -> Float
bias b _ = b

-- Define the instance of Differentiable for Bias
instance Differentiable Bias where
    pD (Bias f) _ = 0
