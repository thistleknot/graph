pre split by # (markdown headers)
	count [double?] new lines between
	ignore 0
	log transform
		chunk [markdown] sections at median + mad * 1.4826 * 2
		to this limit with overlap as median
		reudce_overlap
	chunk within these [deduplicated] sections (track section_idx and chunk_idx, business key)
		recursive text splitter, add \n# to the begging of the custom splitter regex parms we can pass in addition to the defaults
	merge sections up to median (ignore overlaps)
		this ensure the atomic size is the median (min size is median, max size is up to the upper boundary, but even that was split with median overlap, so we get a normal distribution around this center)

hnsw over bpe sparsevec

 then I'd like to draw a community map just like I did for quotes with up to top 3 chunks displayed.  This will help me see what topics are covered and what the medoid and close mmr neighbors are within that given topic
 
❯ oh yeah and I want to avoid ingesting references, but that's easier said than done, but often is separated by other markdown sections which help.  So a section that contains the phrase lower references or citations or 'works cited' or whatever else you think is common, run until the next section else end of document, but this assumes docling can properly interpret the next section if one exists.  We can check for outliers using the same formula we do chunks and have an llm determine if there is another section after the references one.

❯ I'm actually fine with not trying to drop them, but track the section name properly so I can exclude them during retrieval

❯ I feel like since we have n chunks we can figure out the proper 95% prediction interval and chunk at this size, but I think that's overkill.  If median stats stand by normalized stats with sufficient n, then I see no reason to over compensate, we already know we have a large n to observe measures of central tendency (median, and median absolute deviation)+