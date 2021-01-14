const fs = require("fs").promises;
const csv = require('async-csv');
const shuffle = require('fisher-yates');

let TrieSymbolic = require("../src/TrieSymbolic");
let TrieSymbolicPatterns = require("../src/TrieSymbolicPatterns");

// EXPERIMENT PARAMETERS
const numTrials = 10;


describe.skip("5.a.S Access tests for the symbolic trie with symbolic patterns.", () => {
  const fileName = "results/results_5a_symbolic_patterns_";
  
  const numRepeats = 4;
	const numMicrosecondsAccuracy = numRepeats / 1000;
	const accuracy = 0.025; // Unit is Microseconds. = 1000/numRepeats
  
  // Prepares the tests' context.
  const commonSetup = async function (trial) {
    let trieSymbolic = new TrieSymbolic();
	let trieSymbolicPatterns = new TrieSymbolicPatterns();
    
    // Clear the old results.
    await fs.writeFile(`${fileName}${trial}.csv`, "");
    
    // Read the known dictionary file from disk.
    let csvString = await fs.readFile(`dictionaries/dictionaryPatterns.csv`, 'utf-8');
    let rows = shuffle(await csv.parse(csvString));
	
	let words = rows.slice(-1 * rows.length / 2);
	let nonWords = rows.slice(0, rows.length / 2);
	debugger;
	for (let word of words) {
		trieSymbolic.insert(word[0]);
		trieSymbolicPatterns.insert(word[0], word[2]);
	}

    // Ready for the trials.
    return {
      trieSymbolic: trieSymbolic,
	  trieSymbolicPatterns: trieSymbolicPatterns,
      words: words,
      nonWords: nonWords,
	  numWords: rows.length
    };
  };
  
  // Executes each experiment with all its stimuli.
  const commonProcessing = async function (batch, trieSymbolic, trieSymbolicPatterns, stimuli, numWords) {
    const output = [];
    
    output.push([
      "batchIn", "wordIn", 
      "wordKnownIn", "wordLengthIn", "numWordsIn",
	  "resultPatternsOut", "heightPatternsOut", "resultOut", "heightOut", "timeOut"
    ]);

    // const allTime = performance.now();
	
    for (let stimulus of stimuli) {
      // Initialize the measurement.
      let times = numRepeats;
      const currentTime = performance.now();
      
      // Execute the measurement.
      let searchResults;
      while(times--) {
        searchResults = trieSymbolicPatterns.search(stimulus.value[0], stimulus.value[2]);
      }
      const elapsedTime = Math.round(((performance.now() - currentTime) / numMicrosecondsAccuracy) /
		accuracy) * accuracy;
      
	  let plainSearchResult = trieSymbolic.search(stimulus.value[0]);
	  
      output.push([
        batch, stimulus.value[0], 
        stimulus.isKnown ? "true": "false", stimulus.value[0].length, numWords,
		searchResults.found ? "true": "false", searchResults.height, plainSearchResult.found ? "true": "false", plainSearchResult.height, elapsedTime
      ]);
    }
    // const allElapsedTime = performance.now() - allTime;
    
    // Optional
    // output.push([
      // batch, "TOTAL", 
      // "N/A", "N/A", "N/A", 
      // "N/A", allElapsedTime
    // ]);
    
    await fs.appendFile(`${fileName}${batch}.csv`, await csv.stringify(output),'utf-8');
  }
  
  // Orchestrates each test from the test context.
  const commonFlow = async function () {
    
    for(let i = 0; i < numTrials; i++) {
	  let testContext = await commonSetup(i);
	  
      let stimuli = [...testContext.words];
      stimuli = stimuli.map((stimulus) => {
        return {
          value: stimulus,
          isKnown: true
        };
      });
      let nStimuli = [...testContext.nonWords];
      nStimuli =  nStimuli.map((stimulus) => {
        return {
          value: stimulus,
          isKnown: false
        };
      });
      stimuli = stimuli.concat(nStimuli);
      stimuli = shuffle(stimuli);
      
      await commonProcessing(i, testContext.trieSymbolic, testContext.trieSymbolicPatterns, stimuli, testContext.numWords);
    }
  }
  
  test(`Accessing the symbolic trie with  patterns.`, async () => {
		await commonFlow();
		
		// Automatically passing tests.
		expect(true).toEqual(true);
	});
});