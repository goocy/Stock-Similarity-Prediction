import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

def extractSegment(dataFrame, startIndex, segmentLength):
	startTime = df['date'].iloc[startIndex]
	endTime = startTime + segmentLength
	endIndex = np.searchsorted(df['date'], endTime)
	return(dataFrame[startIndex:endIndex])

def correlateTimeseries(A, B):

	# Convert the time series to relative time
	aDate = A['date'] - A['date'].iat[0]
	bDate = B['date'] - B['date'].iat[0]

	# Prepare indices for matched data points
	datesMatched = np.searchsorted(aDate, bDate)
	l = len(aDate) - 1
	datesMatched[datesMatched > l] = l

	# Iterate over data sets in time series
	c = dict()
	keyword = 'price'
	# Select data according to matched indices
	a = np.array(A[keyword].values)
	aReduced = a[datesMatched]
	bReduced = np.array(B[keyword].values)
	# Correct to the baseline
	aReduced = aReduced - np.mean(aReduced)
	bReduced = bReduced - np.mean(bReduced)
	# Perform the z-transformation
	zA = aReduced / np.sqrt(np.sum(np.square(aReduced)) / l)
	zB = bReduced / np.sqrt(np.sum(np.square(bReduced)) / l)
	# Calculate the correlation
	r = pearsonr(zA,zB)
	return r[0]

# Read the example CSV
csvFilename = '/Users/goocy/Downloads/bitfinexUSD.csv.gz'
columnNames = ['date','price','volume']
importFilter = ['date','price']
columnTypes = {'date': np.uint32, 'price': np.float32, 'volume': np.float32}
print('Reading csv file...')
df = pd.DataFrame(pd.read_csv(csvFilename, names = columnNames, dtype = columnTypes, usecols=importFilter))
print('...finished.')

# See how much memory we're using
memUsage = sum(block.values.nbytes for block in df.blocks.values())

# Convert to human-readable date format
# df['date'] = pd.to_datetime(df['date'], unit='s') # Increases 'date' memory usage by 50%

# Convert to logarithmic values
df['price'] = np.log(df['price'])

# Generate a data template
comparisonLength = 3600 # Seconds
templateStartIndex = int(len(df) * 0.6)
template = extractSegment(df, templateStartIndex, comparisonLength)

# Sweep the existing data series for a match with the template
maxLen = len(df) - comparisonLength - 1
maxLen = 5000
similarity = np.zeros([maxLen,])
xScale = xrange(maxLen)
previousPercentage = 0
print('Starting correlation...')
for t in xScale:
	segment = extractSegment(df, 11000+t, comparisonLength)
	percentage = int(np.floor(100 * float(t) / maxLen))
	if percentage > previousPercentage:
		print('{}%'.format(percentage))
		previousPercentage = percentage
	similarity[t] = correlateTimeseries(segment, template)

plt.plot(xScale, np.abs(similarity), 'r')
plt.show()