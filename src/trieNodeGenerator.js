let TrieStats = require("./trieStats.js");

module.exports = class TrieNodeGenerator {
	
  // Initialize the trieNode properties
  constructor(trieStats, prefix, level, index, parent) {
    // Stats
    trieStats.nodes.push(this);
    trieStats.trieNodeCounter++;
    this.Id = trieStats.trieNodeCounter++;
    
    // Trie properties
    // Initialize Children: [Null,...Null, Null]
    this.children = [];
    for (let i = 0; i < TrieStats.ALPHABET_SIZE; i++) {
      this.children[i] = null;
    }
    // Complete words
    this.isEndOfWord = false;
    
    // Optional
    this.prefix = prefix;
    this.level = level;
    this.index = index;
    this.parent = parent;
  }
}