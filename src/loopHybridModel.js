let HybridModel = require("./hybridModel.js");

module.exports = class LoopHybridModel {
  
  constructor(args) {
	this.args = args;
  }
  
  async  Run() {
	var self = this;
					
	return (await (new HybridModel({
		model: {
			sequence: "LoopSequence",
			method:  "LoopMethod",
			argument:  null,
			result: null,
		},
		stack: self.args.stack,
		result: self.args.result,
		symbolicWrapper: self.args.symbolicWrapper,
		symbolicFunctions: {
			getFalse: async function (stack, args) {return false;},
			getCallbackResult: async function (stack, args) {
				// args = null;
				return await self.args.loopCallback();
			}
		}
	})).Run());
  }  
}