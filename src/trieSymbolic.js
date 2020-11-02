let TrieNode = require("./trieNode.js");

module.exports = class TrieSymbolic {
	
  // Initialize the trie properties
  constructor() {
    this.root = new TrieNode();
  }
  
  insert(key) {
    let index;
    
    // Select the root as the Starting state
    let currentNode = this.root;
    
    // Loop over the word
    for (let letter of key) {
      // Gets the letter index
      index = letter.charCodeAt() - 'a'.charCodeAt();
      if (index < 0 || index > 25) {console.log("Alphabetic index problem: " + key);} 
      
      // Create empty TrieNode if there isn't one
      if (currentNode.children[index] == null) {
        currentNode.children[index] = new TrieNode();
      }
      
      // Assign the current trie state to the child node
      // NOTE: Can this performance be optimized? Not with a non-local design!!
      currentNode = currentNode.children[index];
    }
    
    // Capture end properties on the node 
    currentNode.isEndOfWord = true;
  }

  search(key) {
    let index;
    let height = 0;
    
    // Initiates the search
    let currentNode = this.root;
    
    for (let letter of key) {
      // Gets the letter's alphabetic index, a numeric value.
      index = letter.charCodeAt() - 'a'.charCodeAt();
      if (index < 0 || index > 25) {console.log("Alphabetic index problem: " + key);} 
      
      // Assign the current trie state to the child node.
      // Included in timing.
      currentNode = currentNode.children[index];
      
      // Report failure to find an entry.
      if (currentNode == null) {
        break
      }
      
      height++;
    }
    
    // The node for the last letter is found AND
    // The node that is found is marked as the end of the word
    let result = currentNode != null && currentNode.isEndOfWord;
    
    return {
      found: result,
      height: height
    };
  }
}