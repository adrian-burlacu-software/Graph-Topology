const { Worker } = require('worker_threads');
const flatPromise = require("flat-promise");

module.exports = class MultiThreadedEngine {

  constructor(args) {
	  // args.numLetters
	  // args.numWords
	  // args.numProcessors
	  this.args = args;
	  
	  this.processes = new Set();
	  this.freeProcesses = [];
  }
  
  async init() {
	for(let i=0; i<this.args.numProcessors; i++) {
		let process = new Worker("./src/worker.js", { workerData: this.args });
		this.processes.add(process);
		this.freeProcesses.push(process);
		process.resultPromise = flatPromise();
		
		process.on('message', (message) => {
			process.resultPromise.resolve(message);
		});
	}
  }
  
  async getResult(args) {
    let process = this.freeProcesses.pop();
	let result;

	process.postMessage(args, [args.buffer]);
	result = await process.resultPromise.promise;

	// Cleanup
	process.resultPromise = flatPromise();
	this.freeProcesses.push(process);
	
	return result;
  }
  
  async close() {
	for(let process of this.processes) {
		process.terminate();
	}
  }
}
