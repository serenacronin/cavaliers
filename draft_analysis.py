# set up the probability to be different depending on 
# the S/N of the pixel
if max(spectrum) <= median_amp:
    prob = 0.99999
elif (max(spectrum) > median_amp) & (max(spectrum) < 3*median_amp):
    prob = 0.999999
elif (max(spectrum) > 3*median_amp) & (max(spectrum) < 6*median_amp):
    prob = 0.9999999
elif (max(spectrum) > 6*median_amp) & (max(spectrum) < 9*median_amp):
    prob = 0.99999999
elif (max(spectrum) > 9*median_amp) & (max(spectrum) < 12*median_amp):
    prob = 0.999999999
elif max(spectrum) >= 12*median_amp:
    prob = 0.9999999999
    
# reduced chi square upper limit from the chisquare distribution
upperlim = sp.stats.chi2.ppf(prob, df=len(spectrum)-free_params1) / (len(spectrum)-free_params1)

	# FIXME: MOVE TO ANALYSIS
	# # calculate the median amplitude of the cube; to be used later
	# # then save as pkl file; if this file already exists
	# # the just open it
	# if not os.path.exists('%s/median_amp.pkl' % savepath):
	# 	median_amp = calculate_median_amplitude(chunk, chunk_num, multiprocess)
	# 	pickle.dump(median_amp, open('%s/median_amp.pkl' % savepath, 'wb'))
	
	# else:
	# 	median_amp = pickle.load(open('%s/median_amp.pkl' % savepath, 'rb'))