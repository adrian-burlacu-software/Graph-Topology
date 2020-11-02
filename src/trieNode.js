let TrieStats = require("./trieStats.js");

module.exports = class TrieNode {
	
  // Initialize the trieNode properties
  constructor() {
    
    // Trie properties
    // Initialize Children: [Null,...Null, Null]
    this.children = new Array(TrieStats.ALPHABET_SIZE).fill(null);

    // Complete words
    this.isEndOfWord = false;
  }
}