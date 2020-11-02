// Stores constants and stats for the trie.
module.exports = class TrieStats {
  static ALPHABET_SIZE = 26;
  
  // Initialize the TrieStats properties
  constructor() {
    this.trieNodeCounter = -1;
    this.nodes = [];
  }
}