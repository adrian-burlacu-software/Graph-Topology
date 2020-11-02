const { parentPort, workerData } = require('worker_threads');
const Struct = require('struct');

const chunk = (arr, size) =>
	  Array.from({ length: Math.ceil(arr.length / size) }, (v, i) =>
		arr.slice(i * size, i * size + size)
	  );
	  
let decoder = new TextDecoder();

parentPort.on('message', (message) => {
	let result = {
		count: 0,
		nodesToProcess: new Map(),
		areWords: new Map()
	};
	let resultMap = new Map();
	let resultBuffers = [];

	for(let word of chunk(new Uint8Array(message.buffer), workerData.numLetters)) {
		word = decoder.decode(word);
		
		let earlyEnding = word.indexOf('\0');
		if (earlyEnding > -1) {
			word = word.substr(0, earlyEnding);
		}
		if (!word) {break;}
		
		let letter = word[0];
		let destination = resultMap.get(letter);
		
		let out = '';
		if (!destination) {
			destination = [];
			resultMap.set(letter, destination);
		}
		
		let detachedWord = word.substring(1);
		if (!detachedWord) {
			result.areWords.set(letter, true);
			continue;
		}
		
		destination.push(detachedWord);
		result.count = Math.max(result.count, destination.length);
	}
	
	//console.log(resultMap);
	//console.log(result.areWords);
	
	for (let [key, value] of resultMap.entries()) {
		let resultStruct = Struct().array('items', result.count, 
											'chars',  workerData.numLetters - 1);
		resultStruct.allocate();
		let buffer = resultStruct.buffer().buffer;
		result.nodesToProcess.set(key, buffer);
		resultBuffers.push(buffer);
		
		let counter = 0;
		for(let word of value) {
			resultStruct.fields.items[counter++] = word;
		}
	}
	
	parentPort.postMessage(result, resultBuffers);
  });

