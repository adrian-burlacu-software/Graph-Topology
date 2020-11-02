module.exports = class HybridModel {
  
  constructor(args) {
	this.args = args;
	this.args.symbolicFunctions.init = function () {return false;};
	this.args.symbolicFunctions.end = function () {return false;};
  }
  
  getMethod(methodName) {
	return this.args.symbolicFunctions[methodName];
  }
  
  async Run() {
	this.currentStep = 0;
	this.resultError = false;
	
	while (true) {
		this.currentStep = await this.args.symbolicWrapper.Get([this.args.model.sequence, this.currentStep, this.resultError]);
		this.method = await this.args.symbolicWrapper.Get([this.args.model.method, this.currentStep]);
		this.argument = this.args.model.argument ? await this.args.symbolicWrapper.Get([this.args.model.argument, this.currentStep]) : "[]";
		this.result = this.args.model.result ? await this.args.symbolicWrapper.Get([this.args.model.result, this.currentStep]) : "resultError";
		
		if (this.currentStep == -1) {
			break;
		}
				
		if (this.result == "resultError") {
			this.resultError = await (this.getMethod(this.method))(this.args.stack, JSON.parse(this.argument));
		} else {
			this.args.stack[this.result] = await (this.getMethod(this.method))(this.args.stack, JSON.parse(this.argument));
		}
		
		if (this.method == "getNextNodeId" || this.args.stack[2] == "2") {
			//debugger;
		}
		
		if (this.resultError == true) {
			break;
		}
	}
	
	return {
		result: this.args.stack[this.args.result],
		resultError: this.resultError
	};
  }  
}