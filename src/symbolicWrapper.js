// Stores constants and stats for the trie.
module.exports = class SymbolicWrapper {
  
  constructor(javaEngine) {
	  this.javaEngine = javaEngine;
  }
  
  async Get(args) {

	// Uses a switch model to prevent exceeding the maximum number of representable concepts by the neural network
	switch (args[0]) {
		case "TrieModel4":
			args[0] = args[0] + (await this.javaEngine.getResult(["TrieModelSwitch", args[1]])).trim();
			break;
		case "TrieCheck4":
			args[0] = args[0] + (await this.javaEngine.getResult(["TrieEndSwitch", args[1]])).trim();
			break;
		case "TrieEnd4":
			args[0] = args[0] + (await this.javaEngine.getResult(["TrieEndSwitch", args[1]])).trim();
			break;
		default:
			break;
	}
	
	for (var i = 0; i < args.length; i++) {
	  switch (args[i]) {
		case false:
			args[i] = "FALSE";
			break;
		case true:
			args[i] = "TRUE";
			break;
		default:
			 args[i] = args[i] + "";
			break;
	  }
	}
	
	let result = (await this.javaEngine.getResult(args)).trim();
	
	if(result === 'ERROR') {
		throw new Error();
	}
	
	// Gets the result from the ML model
	return result;
  }
}
