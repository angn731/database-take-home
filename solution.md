
## Solution 

### Progress by Two-Hour Mark

I accidentally tried to push all my files to remote when I was making commits during the assessment, but the initial results and optimized results files were too large to push. I didn't realize this until after the two hours were up, however. I have all the commits saved locally, but unfortunately, they did not push to remote. 

Here is my local git log for reference:
______________________________________________________________
commit e347ae626c6c303f5bcb4c346f786f1af4afaebf (HEAD -> main)
Author: Angelina <ning.angelina@gmail.com>
Date:   Thu Mar 13 15:27:06 2025 -0400

    Track large files with Git LFS

commit 546c53ba62f503ff58aefb02ee8ed584faf62dcf
Author: Angelina <ning.angelina@gmail.com>
Date:   Thu Mar 13 15:23:54 2025 -0400

    Implemented a BFS search to add connections to highly queried nodes

commit c017b59bf05ce70cc358a5ae8c351cdd47dc3627
Author: Angelina <ning.angelina@gmail.com>
Date:   Thu Mar 13 14:30:56 2025 -0400

    Reassigned weights based on how often a neighboring node is targeted. Pruned edges of nodes based on ratio of sum of edge w
eights / # edges

commit 2c13723bf7cc5453536e25f32dd8499b1074fc57 (origin/main, origin/HEAD)
______________________________________________________________

I completed my coding in 2 hours, but had a meeting scheduled right after I completed the coding, so I was unable to complete this Solutions writeup until later. 

### Approach & Analysis

I looked up which targets could not be found in the initial graph. I then looked up these targets nodes in the initial graph and found that for the targets the random walk query was unable to find, there were very few edges (sometimes 0 or 1) connecting to those targets from other nodes. 

### Optimization Strategy

My strategy was to identify the most frequented queries, then reweight existing connections to prioritize frequently targeted queries over less targeted or never targeted nodes. I also wanted to identify nodes far from frequently queried targets in the initial graph and add edges to those nodes to improve the likelihood of reaching target nodes from those nodes. Through implementing these two steps, I hoped to reassign weights to edges based on the importance of those edges (evaluated based on their connections to target nodes and the frequency with which those targets are queried). By assigning weights in this manner, I could then prune edges with lower weights to ensure I did not exceed the maximum number of allowed edges per graph. 

### Implementation Details

I decided to iterate through all the nodes of the initial graph and reassign weights to existing edges based on which edges connected to target nodes. If an edge connected to a target node, I weighted the edge proportional to the frequency with which the target node is queried. I also implemented an BFS to determine the shortest distance from each node to a target node. 

After that, I sorted the nodes from longest shortest distance to a target node to shortest. I then assigned an edge from each of these nodes to a target node, with the target node chosen being proportional to its query frequency. 

Finally, I implemented a pruning function to make sure the number of edges in the graph didn't exceed 1000. I iterated through each node to make sure a given node didn't have more than 3 edges, and if it did, I removed edges with the lowest weights (I previously weighted edges based on connections to target nodes, so I knew these edges were of lowest importance). I also calculated a (sum of edge weights) / (number of edges) ratio for each node. If I needed to remove more edges after individually pruning each node, I chose to begin pruning from nodes with the lowest such ratio, since I judged these to be the least important. 

### Results

SUCCESS RATE:
  Initial:   78.5% (157/200)
  Optimized: 93.5% (187/200)
  ✅ Improvement: 15.0%

PATH LENGTHS (successful queries only):
  Initial:   546.0 (157/200 queries)
  Optimized: 189.0 (187/200 queries)
  ✅ Improvement: 65.4%

COMBINED SCORE (success rate × path efficiency):
  Score: 220.48
  Higher is better, rewards both success and shorter paths

### Trade-offs & Limitations

While adjusting edge weights based on query frequency can improve access to popular nodes, it can also lead to overemphasis on frequently queried nodes. Nodes that aren't frequently queried might become isolated. 

Also, if given more time, I would not have implemented BFS search on every node. BFS takes linear time, with regards to the number of vertices and the number of edges in a graph, for each node. Doing it across all nodes takes polynomial time. I would have instead connected a source node to all target nodes, then determined the shortest distance from the source node to each node in the graph. I would have then subtracted one from these distances to get the shortest distance from each node in the graph to a target node. 

### Iteration Journey

I chose to emphasize connectivity to frequently queried nodes in my approach since success rate was the primary driver of the combined score. With this in mind, I decided to not assign any new edges initially and to simply reassign weights to existing edges based on connections to targeted nodes and the query frequency of those nodes. For edges connecting nodes to target nodes, I decided to scale the weight between 5 and 10 based on the query frequency of a target node. For edges connecting nodes to non-target nodes, I scaled the weight to be between 0 and 5. 

I then had to prune the graph. I initially thought about removing the lowest weight edges in the graph (given that removing an edge would not cause a node to become isolated), but felt this too be to "random". Instead, I chose to calculate a (sum of edge weights) / (number of edges) ratio for each node and prune edges of nodes with the lowest ratios. I chose this metric as a proxy for how "important" a node in the graph was, but looking back, this could isolate nodes and may not have been the best choice. 

I then tested the performance of my graph and found that there was slight improvement. This was when I realized it would be a good idea to identify which target nodes could not be found and look at the paths taken in the initial graph to identify the cause. I realized that very few edges connected to target nodes that could not be found. I wanted to add more edges to these target nodes, but had to decide which other nodes to connect target nodes to. 

Since the random walk query works by selecting a random start node, I wanted to establish more connections between "isolated" nodes (nodes far from target nodes) and target nodes, in case the query chose to start from an "isolated" node. This was why I decided to implement a BFS algorithm—I ended up spending much of the last half hour debugging the BFS algorithm and the rest of my optimize graph function because I ran into Key Errors. Even with print statements, it took some time for me to realize that there were some faulty variable assignments and some keys were strings, while others were integers (the queries were passed in as integers, but the nodes in the graph json file were passed in as strings). 

---

* Be concise but thorough - aim for 500-1000 words total
* Include specific data and metrics where relevant
* Explain your reasoning, not just what you did
