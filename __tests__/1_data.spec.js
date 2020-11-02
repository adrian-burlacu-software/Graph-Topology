const fs = require("fs").promises;
const csv = require('async-csv');
const shuffle = require('fisher-yates');

let TrieData = require("../src/TrieData");
let DataWrapper = require("../src/DataWrapper");

// EXPERIMENT PARAMETERS
const numTrials = 10;

describe.skip("1.a.S Access tests for the data trie.", () => {
  const fileName = "results/results_1a_data_";
  const maxWordLength = 9;
  
  const numRepeats = 25;
  // const numMicrosecondsAccuracy = numRepeats / 1000;
  const accuracy = 0.04; // Unit is Milliseconds.
  
  // Prepares the tests' context.
  const commonSetup = async function (numLetters) {
	let dataWrapper = new DataWrapper();
	await dataWrapper.init();
    let trieData = new TrieData(dataWrapper);
    
    // Clear the old results.
    await fs.writeFile(`${fileName}${numLetters}.csv`, "");
    
    // Read the known dictionary file from disk.
    let csvString = await fs.readFile(`dictionaries/dictionaryN15.csv`, 'utf-8');
    let rows = await csv.parse(csvString);
	let allWords = rows.map((row) => {return row[0];});
    let words = allWords.filter((row) => {return row.length <= numLetters;});
    
    for (let word of words) {
      await trieData.insert(word);
    }
	
	let nonWords = allWords.filter((row) => {return words.indexOf(row) === -1;});
    allWords = words;
	
    // Get the stimulus words.
    csvString = await fs.readFile(`dictionaries/stimulus_H_dictionaryN15.csv`, 'utf-8');
    rows = await csv.parse(csvString);
    words = rows.map((row) => {return row[0];}).filter((row) => {return row.length <= numLetters;});
	
    // Ready for the trials.
    return {
      trieData: trieData,
      words: words,
      nonWords: nonWords,
	  numWords: allWords.length
    };
  };
  
  
  // Executes each experiment with all its stimuli.
  const commonProcessing = async function (numLetters, batch, trieData, stimuli, numWords) {
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
		debugger;
        searchResults = await trieData.search(stimulus.value);
      }
      const elapsedTime = Math.round(((performance.now() - currentTime) / numRepeats) / 
		accuracy) * accuracy;
      
      output.push([
        numLetters, batch, stimulus.value, 
        stimulus.isKnown ? "true": "false", stimulus.value.length, numWords, 
		searchResults.found ? "true": "false", searchResults.height, elapsedTime
      ]);
    }
    // const allElapsedTime = performance.now() - allTime;
    
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
      stimuli = shuffle(stimuli).slice(-1 * stimuli.length / 2); 
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
      
      await commonProcessing(numLetters, i, testContext.trieData, stimuli, testContext.numWords);
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

describe.skip("1.b.S Growth for the data trie.", () => {
    const fileName = "results/results_1b_data_";
	const maxWordLength = 11;
	
    const numRepeats = 25;
	const accuracy = 0.04; // Unit is Milliseconds.
	
  
  // Prepares the tests' context.
  const commonSetup = async function (numLetters) {
    // Clear the old results.
    await fs.writeFile(`${fileName}${numLetters}.csv`, [
			"lettersIn", "batchIn", "wordIn", "wordLengthIn", 
			"numWordsOut", "numReadLettersOut", "numNewLettersOut", "timeOut"
		].join(",") + "\n");
    
    // Read the known dictionary file from disk.
    let csvString = await fs.readFile(`dictionaries/stimulus_H_dictionaryN15.csv`, 'utf-8');
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
  
  // calculates number of nodes without changing more code
  const  sharedStart = function (array){
	  var A = array.concat().sort(), 
		a1 = A[0], 
		a2 = A[A.length-1],
		L = a1.length, i= 0;
	  
	  
	  while(i<L && a1.charAt(i)=== a2.charAt(i)) i++;
	  return a1.substring(0, i);
  }
  
  // Executes each experiment with all its stimuli.
  const commonProcessing = async function (numLetters, batch, stimuli) {
    const output = [];
    let trieDatas = [];
    let times = numRepeats;
    while(times--) {
	  let dataWrapper = new DataWrapper();
	  await dataWrapper.init();
      trieDatas[times-1] = new TrieData(dataWrapper);
    }
    
    //const allTime = performance.now();
	let numStimulus = 0;
	let previousStimulus = "";
    for (let stimulus of stimuli) {
      // Initialize measurement.
      times = numRepeats;
      const currentTime = performance.now();
      
      for(let trieData of trieDatas) {
        await trieData.insert(stimulus);
      }
      
      const elapsedTime = Math.round(((performance.now() - currentTime) / numRepeats)
		/ accuracy) * accuracy;
	  const numReadLettersOut = sharedStart([previousStimulus, stimulus]).length;
		
      output.push([
        numLetters, batch, stimulus, stimulus.length,
        numStimulus, numReadLettersOut, stimulus.length - numReadLettersOut, elapsedTime
      ]);
	  
	  previousStimulus = stimulus;
	  numStimulus++;
    }
    // const allElapsedTime = performance.now() - allTime;
    
    // // Optional
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
      stimuli = shuffle(stimuli).slice(-1 * stimuli.length / 2).sort();
      
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
