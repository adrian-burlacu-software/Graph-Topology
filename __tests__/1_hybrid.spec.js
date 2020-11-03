const flatPromise = require("flat-promise");
const fs = require("fs").promises;
const csv = require('async-csv');
const shuffle = require('fisher-yates');
const prompt = require("prompt-async");

let JavaEngine = require("../src/javaEngine.js");
let SymbolicWrapper = require("../src/symbolicWrapper.js");
let TrieHybrid = require("../src/trieHybrid.js");

// EXPERIMENT PARAMETERS
const numTrials = 10;

describe("1.a.S Access tests for the hybrid ML trie.", () => {
  const fileName = "results/results_1a_hybrid_";
  
  const numRepeats = 2;
  // const numMicrosecondsAccuracy = numRepeats / 1000;
  const accuracy = 0.5; // Unit is Milliseconds.
  
  // Prepares the tests' context.
  const commonSetup = async function (numLetters) {
	let javaEngine = new JavaEngine();
	await javaEngine.init();
	let symbolicWrapper = new SymbolicWrapper(javaEngine);
	// TODO: pass in numLetters
    let trieHybrid = new TrieHybrid(symbolicWrapper, numLetters);
    
    // Clear the old results.
    await fs.writeFile(`${fileName}${numLetters}.csv`, "");
    
    // Get the complementing dictionary of negative stimuli.
    let csvString = await fs.readFile(`dictionaries/dictionaryInverseN${numLetters}.csv`, 'utf-8');
    let nonWords = await csv.parse(csvString);
    nonWords = nonWords.map((row) => {return row[0];});
	
    // Get the stimulus words.
    csvString = await fs.readFile(`dictionaries/stimulus_H_dictionaryN${numLetters}.csv`, 'utf-8');
    rows = await csv.parse(csvString);
    let words = [];
    for (let word of rows) {
      words.push(word[0]);
    }
    
    // Ready for the trials.
    return {
      trieHybrid: trieHybrid,
      words: words,
      nonWords: nonWords,
	  numWords: numLetters === 3 ? 247 : 911
    };
  };
  
  // Executes each experiment with all its stimuli.
  const commonProcessing = async function (numLetters, batch, trieHybrid, stimuli, numWords) {
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
        searchResults = await trieHybrid.search(stimulus.value);
      }
      const elapsedTime = Math.round(((performance.now() - currentTime) / numRepeats)
		/ accuracy) * accuracy;
	  
      output.push([
        numLetters, batch, stimulus.value,
        stimulus.isKnown ? "true": "false", stimulus.value.length, stimuli.length, 
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
      
      await commonProcessing(numLetters, i, testContext.trieHybrid, stimuli, testContext.numWords);
    }
	
	await testContext.trieHybrid.symbolicWrapper.javaEngine.close();
  }
  
  test("Accessing the graph of words with three letters.", async () => {
    const numLetters = 3;
    await commonFlow(numLetters);
    
    // Automatically passing tests.
    expect(true).toEqual(true);
  });
  
  test("Accessing the graph of words with four letters.", async () => {
    const numLetters = 4;
    await commonFlow(numLetters);
    
    // Automatically passing tests.
    expect(true).toEqual(true);
  });  
});
