const fs = require('fs').promises;
const csv = require('async-csv');
const flatPromise = require("flat-promise");

// Stores constants and stats for the trie.
module.exports = class DataWrapper {
  
  constructor () {
	  this.models = new Map();
	  this.models.set("TrieModel", new Map([["null,null", "0"]]));
	  this.models.set("TrieCheck", new Map([["null", "null"], ["0", "null"]]));
	  this.models.set("TrieEnd", new Map([["null", "FALSE"], ["0", "FALSE"]]));
  }
  
  async init () {
	// Get the file names.
	let files = await fs.readdir('data/');

	for(let file of files) {
		let csvString = await fs.readFile(`data/${file}`, 'utf-8');
		let rows = await csv.parse(csvString);
		
		let modelName = file
				.split("").reverse().join('')
				.substring(4)
				.split("").reverse().join('');
		let model = new Map();
		this.models.set(modelName, model);
		
		let numInputs = 0;
		for (let column of rows[0]) {
			if (column.endsWith('In')) {
				numInputs++;
			} else {
				break;
			}
		}
		
		for (let rowIndex = 1; rowIndex < rows.length; rowIndex++) {
			let row = rows[rowIndex];
			model.set(row.slice(0, numInputs).join(','), row[numInputs]);
		}
	}
  }
  
  async Get(args) {
	
	for (var i = 0; i < args.length; i++) {
	  switch (args[i]) {
		case false:
			args[i] = 'FALSE';
			break;
		case true:
			args[i] = 'TRUE';
			break;
		default:
			 args[i] = args[i] + '';
			break;
	  }
	}

	let model = this.models.get(args[0]);
	let result = model.get(args.slice(1, args.length).join(','));
	
	if(!result) {
		throw new Error();
	}
	
	// Gets the result from the data model
	return result;
  }
  
  async Set(args) {
	for (var i = 0; i < args.length; i++) {
	  switch (args[i]) {
		case false:
			args[i] = 'FALSE';
			break;
		case true:
			args[i] = 'TRUE';
			break;
		default:
			 args[i] = args[i] + '';
			break;
	  }
	}
	
	let model = this.models.get(args[0]);
	model.set(args.slice(1, args.length - 1).join(','), args[args.length - 1]);
  }
}
