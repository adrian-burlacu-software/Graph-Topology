let TrieStats = require("./trieStats.js");
let TrieNodeGenerator = require("./trieNodeGenerator.js");

module.exports = class TrieSymbolicGenerator {
	
  // Initialize the trie properties
  constructor() {
    this.trieStats = new TrieStats();
    this.root = new TrieNodeGenerator(this.trieStats);
  }
  
  insert(key) {
    let height;
    let length = key.length;
    let index;
    
    // Select the root as the Starting state
    let currentNode = this.root;
    
    // Loop over the word
    for (height = 0; height < length; height++) 
    {
      // Gets the letter index
      index = (function (c) { 
        return c.charCodeAt == null ? c : c.charCodeAt(); 
      })(key.charAt(height)) - 'a'.charCodeAt();
      if (index < 0 || index > 25) {console.log("Alphabetic index problem: " + key);} 
      
      // Create empty TrieNode if there isn't one
      if (currentNode.children[index] == null) {       
        currentNode.children[index] = new TrieNodeGenerator(this.trieStats, key.substring(0, height), height, index, currentNode);
      }
      
      // Assign the current trie state to the child node
      currentNode = currentNode.children[index];
    }
    
    // Capture end properties on the node 
    currentNode.isEndOfWord = true;
  }
}