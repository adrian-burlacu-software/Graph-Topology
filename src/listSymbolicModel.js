module.exports = class ListSymbolicModel {
  
  constructor(args) {
	this.args = args;
	this.iterator = args[Symbol.iterator]();
  }
  
  next() {
	
	let theChar = this.iterator.next();
	if (theChar && !theChar.done && theChar.value !== ' ') {
		let result = theChar.value;
		return result;
	}
	else {
		return null;
	}
  }
}