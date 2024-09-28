{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances #-}

module Main where

import Layers
import qualified Data.ByteString.Lazy as BL
import Data.Binary.Get
import Data.Int
import Data.Word
import Text.Printf
import Data.List (foldl')
import Numeric.LinearAlgebra

-- Define the Digit data type
data Digit = Digit { pixels :: [Double], label :: Int }
    deriving (Show)

main :: IO ()
main = do
    -- File paths for the training dataset
    let trainImagesFile = "/Users/blakewhitmer/Downloads/archive/train-images.idx3-ubyte"
    let trainLabelsFile = "/Users/blakewhitmer/Downloads/archive/train-labels.idx1-ubyte"
    let testImagesFile = "/Users/blakewhitmer/Downloads/archive/t10k-images.idx3-ubyte"
    let testLabelsFile = "/Users/blakewhitmer/Downloads/archive/t10k-labels.idx1-ubyte"

    trainDigits <- loadData trainImagesFile trainLabelsFile
    testDigits <- loadData testImagesFile testLabelsFile

    printDigits 5 trainDigits

    printDigits 1 testDigits

    -- Initialize weights and biases
    let w1 = initializeInputWeight
    let w2 = initializeHiddenWeight
    let b1 = initializeHiddenBias
    let b2 = initializeOutputBias

    -- Train the neural network
    let epochs = 5
    let (epochLosses, trainedW1, trainedW2, trainedB1, trainedB2) = trainNNEpochs epochs trainDigits w1 w2 b1 b2

    -- Print loss information
    mapM_ (\(epoch, losses) -> putStrLn $ "Epoch " ++ show epoch ++ " average loss: " ++ show (average losses)) (zip [1..] epochLosses)

    -- You can now use 'trainDigits' to train your neural network
    
-- Function to evaluate the neural network on a list of digits
evaluateNN :: [Digit] -> Weights -> Weights -> Biases -> Biases -> Double
evaluateNN digits w1 w2 b1 b2 =
    let total = length digits
        correct = length $ filter (isCorrect w1 w2 b1 b2) digits
    in (fromIntegral correct) / (fromIntegral total) * 100.0  -- Accuracy in percentage

-- Helper function to check if the network's prediction is correct
isCorrect :: Weights -> Weights -> Biases -> Biases -> Digit -> Bool
isCorrect w1 w2 b1 b2 (Digit pixels label) =
    let x = fromList pixels
        (logits, _, _) = feedForward (x, w1, w2, b1, b2)
        prediction = maxIndex (softmax logits)
    in prediction == label
average :: [Double] -> Double
average xs = sum xs / fromIntegral (length xs)

-- Function to perform a single training step and collect loss
trainStep :: (Weights, Weights, Biases, Biases, [Double]) -> Digit -> (Weights, Weights, Biases, Biases, [Double])
trainStep (w1, w2, b1, b2, losses) (Digit pixels label) =
    let x = fromList pixels
        (neww1, neww2, newb1, newb2, loss) = stepNN (x, label, learningRate, w1, w2, b1, b2)
    in (neww1, neww2, newb1, newb2, loss : losses)

-- Function to train the neural network and collect losses
trainNN :: [Digit] -> Weights -> Weights -> Biases -> Biases -> ([Double], Weights, Weights, Biases, Biases)
trainNN digits w1 w2 b1 b2 =
    let (finalW1, finalW2, finalB1, finalB2, losses) =
            foldl' trainStep (w1, w2, b1, b2, []) digits
    in (reverse losses, finalW1, finalW2, finalB1, finalB2)

-- Function to train the neural network over multiple epochs and collect losses
trainNNEpochs :: Int -> [Digit] -> Weights -> Weights -> Biases -> Biases -> ([[Double]], Weights, Weights, Biases, Biases)
trainNNEpochs 0 _ w1 w2 b1 b2 = ([], w1, w2, b1, b2)
trainNNEpochs n digits w1 w2 b1 b2 =
    let (losses, w1', w2', b1', b2') = trainNN digits w1 w2 b1 b2
        (allLosses, finalW1, finalW2, finalB1, finalB2) = trainNNEpochs (n - 1) digits w1' w2' b1' b2'
    in (losses : allLosses, finalW1, finalW2, finalB1, finalB2)

-- The code below is for formatting data

-- Define data types for image and label headers
data ImageHeader = ImageHeader
    { magicNumber :: Int
    , numItems :: Int
    , numRows :: Int
    , numCols :: Int
    } deriving (Show)

data LabelHeader = LabelHeader
    { magicNumberLabel :: Int
    , numItemsLabel :: Int
    } deriving (Show)

-- Parse the image header
parseImageHeader :: Get ImageHeader
parseImageHeader = do
    magic <- getInt32be
    numImages <- getInt32be
    numRows <- getInt32be
    numCols <- getInt32be
    return $ ImageHeader (fromIntegral magic) (fromIntegral numImages) (fromIntegral numRows) (fromIntegral numCols)

-- Parse the label header
parseLabelHeader :: Get LabelHeader
parseLabelHeader = do
    magic <- getInt32be
    numLabels <- getInt32be
    return $ LabelHeader (fromIntegral magic) (fromIntegral numLabels)

-- Function to split ByteString into chunks of n bytes
chunksOf' :: Int64 -> BL.ByteString -> [BL.ByteString]
chunksOf' n = go
  where
    go bs
        | BL.null bs = []
        | otherwise = let (chunk, rest) = BL.splitAt n bs
                      in chunk : go rest

-- Parse image and label data into a list of Digits
parseData :: BL.ByteString -> BL.ByteString -> Int -> [Digit]
parseData imageData labelData numItems =
    let imageChunks = chunksOf' 784 imageData
        labelList = BL.unpack labelData
        pixelLists = map (map (\w -> fromIntegral w / 255.0) . BL.unpack) imageChunks
        labels = map fromIntegral labelList
    in zipWith Digit pixelLists labels

-- Load data from image and label files
loadData :: FilePath -> FilePath -> IO [Digit]
loadData imageFile labelFile = do
    imageContent <- BL.readFile imageFile
    labelContent <- BL.readFile labelFile

    -- Parse the headers
    let imageHeader = runGet parseImageHeader imageContent
    let labelHeader = runGet parseLabelHeader labelContent

    -- Extract the data after the headers
    let imageData = BL.drop 16 imageContent
    let labelData = BL.drop 8 labelContent

    -- Get the number of images and labels
    let numImages = numItems imageHeader
    let numLabels = numItemsLabel labelHeader

    print numImages
    print numLabels

    -- Ensure the number of images and labels match
    if numImages /= numLabels
        then error "Number of images and labels do not match"
        else do
            -- Parse the data into a list of Digits
            let digits = parseData imageData labelData numImages
            return digits

-- Function to print a single Digit
printDigit :: Digit -> IO ()
printDigit (Digit pixels label) = do
    let rows = chunksOf 28 pixels
    mapM_ printRow rows
    putStrLn $ "Label: " ++ show label
    putStrLn "-------------------------"

-- Helper function to print a single row of pixels
printRow :: [Double] -> IO ()
printRow row = do
    let formattedRow = unwords $ map formatPixel row
    putStrLn formattedRow

-- Function to format a pixel value
formatPixel :: Double -> String
formatPixel 0.0 = "    "  -- Four spaces to align with formatted numbers
formatPixel v   = printf "%.2f" v

-- Function to split a list into chunks of n elements
chunksOf :: Int -> [a] -> [[a]]
chunksOf _ [] = []
chunksOf n xs =
    let (first, rest) = splitAt n xs
    in first : chunksOf n rest

-- Function to iterate over a list of digits and print each one
printDigits :: Int -> [Digit] -> IO ()
printDigits n digits = mapM_ printDigit (take n digits)