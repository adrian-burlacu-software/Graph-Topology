const Struct = require('struct');
const MultiThreadedEngine = require("./multiThreadedEngine.js");

// do width first
// Only parallelize the other words in the current parent,
// to bring along to the next letter. Subtract from current words available.
// If the link exists, don't do that step. This won't be necessary because if I activate a word, it starts from its branching point.
// Words are in pre-determined order, letters are not in order.  just skip words that start with visited letters because their thread will pick them up;
// Could distribute each letter further, but there are diminishing returns.
// Could distribute by letter over nodes regardless of  prefix, but that would perform the access procedure too many times because they would not be organized.
// WAIT SOME TIME AND GET THE INSERT RESULTS...
const NUM_WORDS_INIT = 1000;
const NUM_PROC = 4;

module.exports = class TrieLocal {
	
	
  // Initialize the trie properties
  constructor (numLetters, words) {
	this.multiThreadedEngine;
	this.trieStatsResult = new Map();
	this.numLettersInit = numLetters;
	this.maxWordsPerNode = NUM_WORDS_INIT;
	
	this.nodes = [];
	this.nodes[0] = [];
	this.root = {
		isWord: false,
		children: new Map(),
		parent: null,
		letter: null
		};
	
	for (let word of words) {
		this.insert(word);
	}
  }

  // Private! Not profiled!
  // Prepares the first layer of the router synchronously.
  insert (key) {
	let letterNode = this.root.children.get(key[0]);
	if (!letterNode) {
		let wordsToProcess = Struct().array('items', NUM_WORDS_INIT, 'chars', this.numLettersInit - 1);
		wordsToProcess.allocate();
		letterNode = {
				isWord: false,
				count: 0,
				buffer: wordsToProcess.buffer().buffer,
				wordsToProcess: wordsToProcess,
				children: new Map(),
				parent: this.root,
				letter: key[0]
			};
		
		this.root.children.set(key[0], letterNode);
		this.nodes[0].push(letterNode);
	}

	let result = key.substring(1);
	if (!result) {
		letterNode.isWord=true;
		return;
	}
	
	let destination = letterNode.wordsToProcess.fields.items;
	destination[letterNode.count++] = result;
  }
  
  async init () {
	// High-level array operations that aren't stock somehow...
	const chunk = (arr, size) =>
	  Array.from({ length: Math.ceil(arr.length / size) }, (v, i) =>
		arr.slice(i * size, i * size + size)
	  );
	const zip = (arr1, arr2) => arr1.map((k, i) => [k, arr2[i]]);
	
	// Initialize the measurement.
	let wordsPerNode = this.maxWordsPerNode;
	this.start = performance.now();
	
	for (let numLetters = 1; numLetters < this.numLettersInit; numLetters++) {
		
		// Initialize the worker threads.
		this.multiThreadedEngine = new MultiThreadedEngine({
				numLetters: this.numLettersInit - numLetters,
				numWords: wordsPerNode,
				numProcessors: NUM_PROC
			});
		await this.multiThreadedEngine.init();
		
		// Initialize state for running results on the next layer of the router.
		let wordCounter = 0;
		this.nodes[numLetters] = [];
		
		// Split up the next layer of the router into chunks for the number of workers.
		for (let currentChunk of chunk(this.nodes[numLetters - 1], NUM_PROC)) {
			let chunkPromises = [];

			// Send work to the workers.
			for (let currentNode of currentChunk) {
				chunkPromises.push(this.multiThreadedEngine.getResult({
					buffer: currentNode.buffer					
					}));
			}
			
			
			// Wait for the results to come back from the workers
			let results = await Promise.all(chunkPromises);
			
			// Update the children of each node with the breakdown
			for (let nodeResult of zip(currentChunk, results)) {
				for (let [key, value] of nodeResult[1].nodesToProcess.entries()) {
					
					// Create a new node for the new detached letter. Wow, only the workers read the buffer!
					let newNode = {
							isWord: !!nodeResult[1].areWords.get(key),
							count: nodeResult[1].count,
							buffer: value,
							children: new Map(), 
							parent: nodeResult[0],
							letter: key
						};
					
					// Update the max word count per letter
					wordCounter = Math.max(wordCounter, nodeResult[1].count);
					
					// WOHOO! This line only happens ONCE for every set of associated words.
					nodeResult[0].children.set(key, newNode);
					
					if (newNode.count > 0) {
						this.nodes[numLetters].push(newNode);
					}					
					
					// Save the result without impacting performance
					if (newNode.isWord) {
						// Record each word starting at 2 letters in delta milliseconds.
						this.trieStatsResult.set(newNode, performance.now() - this.start);
					}
				}
			}
		}
		
		this.multiThreadedEngine.close();
		wordsPerNode = wordCounter; // Update the size of worker for the next layer of the router.
	}
  }
  
  getInitStats () {
	  let result = new Map();
	  
	  for (let [key, value] of this.trieStatsResult.entries()) {
		let word = '';
		let currentNode = key;
		
		while (currentNode.parent != null) {
			word += currentNode.letter;
			currentNode = currentNode.parent;
		}
		
		result.set(word.split("").reverse().join(""), value);
	  }
	  
	  return result;
  }

  // TODO
  search(key) {
    let height = 0;
    
    // Initiates the search
    let currentNode = this.root;
    
    for (let letter of key) {
      // Assign the current trie state to the child node.
      // Included in timing.
      currentNode = currentNode.children.get(letter);
      
      // Report failure to find an entry.
      if (currentNode == null) {
        break
      }
      
      height++;
    }
    
    // The node for the last letter is found AND
    // The node that is found is marked as the end of the word
    let result = currentNode != null && currentNode.isWord;
    
    return {
      found: result,
      height: height
    };
  }
}