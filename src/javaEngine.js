const {spawn} = require('child_process');
const flatPromise = require("flat-promise");

module.exports = class JavaEngine {

  constructor() {
  }
  
  async init() {
	this.childProcess = spawn(
	  'java', 
	  ['-cp', '"src;src\\h2o-genmodel.jar"', 'Main'], 
	  {
		  shell: true,
		  stdio: ['pipe', 'pipe',  process.stderr]
	  }
	);
	
	this.promise = flatPromise();
	this.childProcess.stdout.on("data", data => {
	  this.promise.resolve(data + '');
	  this.promise = flatPromise();
	});
	await this.promise.promise;
  }
  
  async getResult(args) {
    this.childProcess.stdin.write(`${args.join(" ")}\n`);
	return (await this.promise.promise);
  }
  
  async close() {
	this.childProcess.stdin.write('End\n');
	return (await this.promise.promise);
  }
}
