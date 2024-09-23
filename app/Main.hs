{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances #-}

module Main where

import qualified Data.ByteString.Lazy as ByteString
import Numeric.LinearAlgebra

main :: IO ()
main = do
    let file = "/Users/blakewhitmer/Downloads/archive/train-images.idx3-ubyte"
    content <- ByteString.readFile file
    putStrLn $ "File size: " ++ show (ByteString.length content) ++ " bytes"
    let strippedcontent = ByteString.drop 16 content
    let firstsixteen = ByteString.take 16 content
    putStrLn $ "File size: " ++ show (ByteString.length strippedcontent) ++ " bytes"
    putStrLn $ "First 16 (bytes): " ++ show (ByteString.unpack firstsixteen)

    printList strippedcontent

    -- Print the array in a more readable format

printList :: ByteString.ByteString -> IO ()
printList bs
    | ByteString.null bs = return ()  -- Base case: stop when ByteString is empty
    | otherwise = do
        let firsttwentyeight = ByteString.take 784 bs
        putStrLn (show (ByteString.unpack firsttwentyeight))  -- Convert to list of bytes for printing
        let restoflist = ByteString.drop 784 bs
        printList restoflist



class Differentiable f where
    pD :: f -> Float -> Float





{-
    pD (relu) z
        | z < 0     = 0  -- Derivative of ReLU when z < 0
        | otherwise = 1  -- Derivative of ReLU when z >= 0
-}


-- This ended up with way more boilerplate than I would like

-- newtype WeightFunction (Weight -> Weight)





{-
weightFunction :: Weight -> Weight
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
-}