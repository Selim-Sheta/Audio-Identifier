
from scipy.io import wavfile
from pathlib import Path
from scipy import signal
import numpy as np
import librosa


# Hop length parameter for calculating the stft
HOP_LENGTH = 256
# FFT length parameter for calculating the stft
N_FFT = 2048
# Threshold controls the detection of peaks in the stft. Must be between 0 and 1.
THRESHOLD = 0.7
# All files are re-sampled at this rate before doing any processing
SAMPL_RATE = 22050
# This controls the number rows of the stft per hash
NUM_FBINS_PER_HASH = 4


def prepareAudio(audioData, fs):
    """ Pre-processing for the audio before building a fingerprint
    Parameters:
        audioData: array containing the audio samples
        fs: sample rate in Hz
    Returns:
        audioData: the monophonic, re-sampled, normalised version of the audio"""

    # Convert to mono
    if audioData.ndim == 2:
        audioData = (audioData[:, 0] + audioData[:, 1]) / 2

    # Re-sample at 22050 Hz to get consistent frequency bins for all recordings
    if fs != SAMPL_RATE:
        audioData = signal.resample(audioData, int(len(audioData) * SAMPL_RATE / fs))

    # Normalise
    if np.amax(np.abs(audioData)) != 0:
        return audioData / np.amax(np.abs(audioData))
    else:
        return audioData


def createFingerprint(signalData, fs):
    """ This function generates a fingerprint from an audio file. It uses librosa to calculate the STFT of the signal,
    and scipy to find the peaks within the STFT. The positions of those peaks are stored in hashes.
    Parameters:
        signalData: array containing the audio samples
        fs: sample rate in Hz
    Returns:
        fingerprint: a numpy array of hashes extracted from the audio"""

    # Pre-processing
    signalData = prepareAudio(signalData, fs)
    # Get the stft of the audio
    specGram = librosa.core.stft(signalData, n_fft=N_FFT, hop_length=HOP_LENGTH)
    # Get total number of frequency bins in the stft
    totalFreqStamps = len(specGram[:, 0])

    # Fingerprint is initialized as an empty list.
    hashes = []
    # Keeps track of the hash index
    hashIndex = 0
    # Loop to generate hashes. Each hash holds
    for stamp in range(0, totalFreqStamps, NUM_FBINS_PER_HASH):
        # add a new empty hash to the fingerprint
        hashes.append([])

        # The hash will hold the contents of the next NUM_FBINS_PER_HASH inverted lists.
        for i in range(NUM_FBINS_PER_HASH):
            # If the current index hasn't exceeded the number of rows in the stft
            if stamp + i < totalFreqStamps:
                # Selecting relevant row of stft
                timeValues = np.abs(specGram[stamp + i])
                # Calculating threshold for peak picking
                thresh = THRESHOLD * np.amax(timeValues)
                # Find positions of the peaks, ignoring first and last 2 values in the list
                peaks = signal.find_peaks(timeValues[2:-2], height=thresh)[0]
                # If some peaks are found
                if peaks != []:
                    # Add their position to current hash
                    hashes[hashIndex].extend(peaks)

        # If hash isn't empty
        if hashes[hashIndex] != []:
            # Remove duplicates
            hashes[hashIndex] = list(dict.fromkeys(hashes[hashIndex]))
        else:
            # -1 indicates empty, useful for saving time later
            hashes[hashIndex] = [-1]

        # Increase hash index
        hashIndex += 1

    # return fingerprint as numpy array
    fingerprint = np.asarray(hashes)
    return fingerprint


def fingerprintBuilder(pathToDataBase, pathToFingerprints):
    """ Creates all the fingerprints for the wav files in the specified path, storing them as .npy files
    Parameters:
        pathToDataBase : path to the database folder
        pathToFingerprints: path to the fingerprint folder"""

    # Retrieving all the wav files in the specified database folder
    files = Path(pathToDataBase).rglob("*.wav")

    # Generating and saving the fingerprint for each file
    for file in files:
        # Reading the wav file
        fs, signalData = wavfile.read(file)

        # Creating fingerprint
        fingerprint = createFingerprint(signalData, fs)

        filePath = str(file)  # Converting path to string
        s = filePath.split("\\")  # Splitting string
        trackName = s[len(s) - 1][:-4]  # Isolating track name
        outfile = pathToFingerprints + '/' + trackName  # Creating the path for the new file

        # Saving the fingerprint as a .npy file
        np.save(outfile, fingerprint, allow_pickle=True)


def matchFunction(queryFingerprint, databaseFingerprint):
    """ Calculates the matching score between two fingerprints, one corresponding to a query and the other to
    a database file.
    Parameters:
        queryFingerprint: fingerprint of the query
        databaseFingerprint: fingerprint of the database file
    Returns:
        score: The highest value in the matching function """

    # Finding largest time stamp in query fingerprint
    maxQueryTStamp = max(max(x) for x in queryFingerprint) + 1
    # Finding largest time stamp in database fingerprint
    maxDataTStamp = max(max(x) for x in databaseFingerprint) + 1
    # If either fingerprint is empty
    if maxQueryTStamp == 0 or maxDataTStamp == 0:
        # score is 0
        return 0

    # Initialising match function as an array of zeros
    matchFunctionArray = np.zeros(maxDataTStamp + maxQueryTStamp - 1)
    # Cycling through the hashes in the query to calculate the match function
    for hashIndex in range(len(queryFingerprint)):
        # If the current hash in either fingerprint is not empty
        if queryFingerprint[hashIndex] != [-1] and databaseFingerprint[hashIndex] != [-1]:
            # Going through all the time stamps in the current hash
            for n in queryFingerprint[hashIndex]:
                # Create shifted list
                currentShiftedList = databaseFingerprint[hashIndex] - n * np.ones(len(databaseFingerprint[hashIndex]))
                # For every item in the shifted list
                for item in currentShiftedList:
                    # increase the corresponding element in match function by 1
                    matchFunctionArray[int(item + maxQueryTStamp)] += 1

    # Return score
    score = np.amax(matchFunctionArray)
    return score


def audioIdentification(querySetPath, fingerPrintPath, outputPath):
    """Performs audio identification on all the wav files in the query path by comparing their fingerprint with
    all the fingerprints stored in the fingerprint folder. The results are stored line by line in the text file
    in the following format: queryaudio1.wav databaseaudiox1.wav databaseaudiox2.wav databaseaudiox3.wav
    where the file names in each line are separated by a tab, and the three database recordings per line are
    the top three ones which are identified by the audio identification system as more closely matching
    the query recording, in order of first to third.
    Parameters:
        querySetPath : path to the database folder
        fingerPrintPath: path to the fingerprint folder
        outputPath: path to the text file in which to write the results"""

    # Prepare output txt file for writing
    textFile = open(outputPath, 'w')
    # Retrieving all the query audio files in the specified folder
    queryFiles = Path(querySetPath).rglob("*.wav")
    # Go through each query in the file
    for query in queryFiles:
        # Loading the audio data
        fs, signalData = wavfile.read(query)
        # Creating query's fingerprint
        queryFingerprint = createFingerprint(signalData, fs)

        # This array contains the matching score for each fingerprint, in the order in which they are read
        matchScoreArray = []
        # This array contains each fingerprint's original track name, in the order in which they are read
        fileNames = []

        matchScoreIndex = 0  # This keeps track of our position in the array of scores
        # Retrieving all the fingerprints in the fingerprint folder
        fingerprintFiles = Path(fingerPrintPath).rglob("*.npy")
        # Go through each fingerprint to calculate the matching score
        for fingerprint in fingerprintFiles:
            # Load the fingerprint as an array of hashes
            currentInvLists = np.load(fingerprint, allow_pickle=True)
            # Calculate and append score to matchScoreArray
            score = matchFunction(queryFingerprint, currentInvLists)
            matchScoreArray.append(score)
            # Increase index
            matchScoreIndex += 1

            filePath = str(fingerprint)                 # Converting path to string
            s = filePath.split("\\")                    # Splitting string
            trackName = s[len(s) - 1][:-4]              # Isolating track name
            fileNames.append(trackName + '.wav    ')    # Add fingerprint's name to fileNames

        # Finding the top 3 best matches
        firstGuess = np.argmax(matchScoreArray)
        matchScoreArray[firstGuess] = 0
        secondGuess = np.argmax(matchScoreArray)
        matchScoreArray[secondGuess] = 0
        thirdGuess = np.argmax(matchScoreArray)

        filePath = str(query)  # Converting path to string
        print("evaluating query : " + filePath + "\n")
        s = filePath.split("\\")  # Splitting string
        queryName = s[len(s) - 1] + '    '  # Isolating query name

        # Adding audio identification results for this query to the output file
        docLine = queryName + fileNames[firstGuess] + fileNames[secondGuess] + fileNames[thirdGuess]
        textFile.write(docLine + '\n')

    # Closing the file
    textFile.close()
