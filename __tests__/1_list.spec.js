const fs = require("fs").promises;
const csv = require('async-csv');
const shuffle = require('fisher-yates');

// EXPERIMENT PARAMETERS
const numTrials = 10;

describe("1.a.S Access for the list.", () => {
  const fileName = "results/results_1a_list_";
  const maxWordLength = 9;
  
  // I was too lazy to model these better...but it works...
  const numRepeats = 8000;
  const numMicrosecondsAccuracy = numRepeats / 1000;
  const accuracy = 0.125; // Unit is Microseconds.
  
  // Prepares the tests' context.
  const commonSetup = async function (numLetters) {
    let trieList = [];
    
    // Clear the old results.
    await fs.writeFile(`${fileName}${numLetters}.csv`, "");
    
    // Read the known dictionary file from disk.
    let csvString = await fs.readFile(`dictionaries/dictionaryN15.csv`, 'utf-8');
    let rows = await csv.parse(csvString);
	let allWords = rows.map((row) => {return row[0];});
    let words = allWords.filter((row) => {return row.length <= numLetters;});
    
    for (let word of words) {
      trieList.push(word);
    }
	
	let nonWords = allWords.filter((row) => {return words.indexOf(row) === -1;});
    allWords = words;
	
    // Get the stimulus words.
    csvString = await fs.readFile(`dictionaries/dictionaryN15.csv`, 'utf-8');
    rows = await csv.parse(csvString);
    words = rows.map((row) => {return row[0];}).filter((row) => {return row.length <= numLetters;});
	
    // Ready for the trials.
    return {
      trieList: trieList,
      words: words,
      nonWords: nonWords,
	  numWords: allWords.length
    };
  };
  
  // Executes each experiment with all its stimuli.
  const commonProcessing = async function (numLetters, batch, trieList, stimuli, numWords) {
    const output = [];
    
    output.push([
      "lettersIn", "batchIn", "wordIn", 
      "wordKnownIn", "wordLengthIn", "numWordsIn",
	  "resultOut", "heightOut", "timeOut"
    ]);

    // const allTime = performance.now();
	
    for (let stimulus of stimuli) {
      // Initialize the measurement.
      let times = numRepeats;
      const currentTime = performance.now();
      
      // Execute the measurement.
      let searchResults;
      while(times--) {
        searchResults = trieList.indexOf(stimulus.value);
      }
      const elapsedTime = Math.round(((performance.now() - currentTime) / numMicrosecondsAccuracy) / accuracy) * accuracy;
      
      output.push([
        numLetters, batch, stimulus.value, 
        stimulus.isKnown ? "true": "false", stimulus.value.length, numWords,
		searchResults > -1 ? "true": "false", searchResults > -1 ? searchResults : numWords, elapsedTime
      ]);
    }
    //const allElapsedTime = performance.now() - allTime;
    
    // Optional
    // output.push([
      // numLetters, batch, "TOTAL", 
      // "N/A", "N/A", "N/A", 
      // "N/A", allElapsedTime
    // ]);
    
    await fs.appendFile(`${fileName}${numLetters}.csv`, await csv.stringify(output),'utf-8');
  }
  
  // Orchestrates each test from the test context.
  const commonFlow = async function (numLetters) {
    let testContext = await commonSetup(numLetters);
    
    for(let i = 0; i < numTrials; i++) {
      let stimuli = [...testContext.words];
      stimuli = shuffle(stimuli).slice(-1 * stimuli.length / 6); 
      stimuli = stimuli.map((stimulus) => {
        return {
          value: stimulus,
          isKnown: true
        };
      });
      let nStimuli = [...testContext.nonWords];
      nStimuli =  shuffle(nStimuli).slice(-1 * stimuli.length).map((stimulus) => {
        return {
          value: stimulus,
          isKnown: false
        };
      });
      stimuli = stimuli.concat(nStimuli);
      stimuli = shuffle(stimuli);
      
      await commonProcessing(numLetters, i, testContext.trieList, stimuli, testContext.numWords);
    }
  }
  
  for (let numLetters = 2; numLetters < maxWordLength; numLetters++) {
	test(`Accessing the graph of words with ${numLetters} letters.`, async () => {
		await commonFlow(numLetters + 1);
		
		// Automatically passing tests.
		expect(true).toEqual(true);
	});
  }
});

describe("1.b.S Growth for the list.", () => {
  const fileName = "results/results_1b_list_";
  const maxWordLength = 11;
  
  const numRepeats = 40000;
  const numMicrosecondsAccuracy = numRepeats /  1000;
  const accuracy = 0.025;  // Unit is Microseconds.
 
  // Prepares the tests' context.
  const commonSetup = async function (numLetters) {
    // Clear the old results.
    await fs.writeFile(`${fileName}${numLetters}.csv`, [
			"lettersIn", "batchIn", "wordIn", 
			"wordLengthIn", "numWordsOut", "timeOut"
		].join(",") + "\n");
    
    // Read the known dictionary file from disk.
    let csvString = await fs.readFile(`dictionaries/dictionaryN15.csv`, 'utf-8');
    let rows = await csv.parse(csvString);
    const words = [];
    
    for (let word of rows) {
      if (word[0].length > numLetters) {
        continue;
      }
      words.push(word[0]);
    }
    
    // Ready for the trials
    return words;
  };
  
  // Executes each experiment with all its stimuli.
  const commonProcessing = async function (numLetters, batch, stimuli) {
    const output = [];
    let trieLists = [];
    let times = numRepeats;
    while(times--) {
      trieLists[times-1] = [];
    }
    
    // const allTime = performance.now();
	let numStimulus = 0;
	
	// Count these
    for (let stimulus of stimuli) {
      // Initialize measurement.
      times = numRepeats;
      const currentTime = performance.now();
      
      for(let trieList of trieLists) {
        trieList.push(stimulus);
      }
      
      const elapsedTime = Math.round(((performance.now() - currentTime) / numMicrosecondsAccuracy) / 
		accuracy) * accuracy;
		
      output.push([
        numLetters, batch, stimulus, 
        stimulus.length, numStimulus, elapsedTime
      ]);
	  
	  numStimulus++;
    }
    // const allElapsedTime = performance.now() - allTime;
    
    // Optional
    // output.push([
        // numLetters, batch, "TOTAL", 
        // "N/A",
        // allElapsedTime
      // ]);
    
    await fs.appendFile(`${fileName}${numLetters}.csv`, await csv.stringify(output),'utf-8');
  }
  
  // Orchestrates each test from the test context.
  const commonFlow = async function (numLetters) {
    let testContext = await commonSetup(numLetters);
    
    for(let i = 0; i < numTrials; i++) {
      let stimuli = [...testContext];
      // Don't forget these are in alphabetical order so that they don't cause conflicts.
      stimuli = shuffle(stimuli).slice(-1 * stimuli.length / 6);
      
      await commonProcessing(numLetters, i, stimuli);
    }
  }
  
  for (let numLetters = 2; numLetters < maxWordLength; numLetters++) {
    test(`Graph genesis of the graph for all words with ${numLetters} letters.`, async () => {
      await commonFlow(numLetters + 1);
      // Automatically passing tests.
      expect(true).toEqual(true);
    });
  }
});
