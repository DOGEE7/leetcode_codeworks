package leetcode.editor;

import leetcode.editor.util.*;

import java.util.*;

//将两个升序链表合并为一个新的 升序 链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的。
//
//
//
// 示例 1：
//
//
//输入：l1 = [1,2,4], l2 = [1,3,4]
//输出：[1,1,2,3,4,4]
//
//
// 示例 2：
//
//
//输入：l1 = [], l2 = []
//输出：[]
//
//
// 示例 3：
//
//
//输入：l1 = [], l2 = [0]
//输出：[0]
//
//
//
//
// 提示：
//
//
// 两个链表的节点数目范围是 [0, 50]
// -100 <= Node.val <= 100
// l1 和 l2 均按 非递减顺序 排列
//
//
// Related Topics 递归 链表 👍 3788 👎 0


//leetcode submit region begin(Prohibit modification and deletion)
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */

class TreeNode {
    int val;
    TreeNode left;
    TreeNode right;

    TreeNode() {
    }

    TreeNode(int val) {
        this.val = val;
    }

    TreeNode(int val, TreeNode left, TreeNode right) {
        this.val = val;
        this.left = left;
        this.right = right;
    }
}

public class Solution {
    public ListNode mergeTwoLists(ListNode list1, ListNode list2) {
        if (list1 == null) {
            return list2;
        }
        if (list2 == null) {
            return list1;
        }
        if (list1.val < list2.val){
            list1.next = mergeTwoLists(list1.next, list2);
            return list1;
        }else {
            list2.next = mergeTwoLists(list1, list2.next);
            return list2;
        }
    }

    public ListNode mergeKLists(ListNode[] lists) {
        ListNode dummy = null;
        ListNode current = dummy;

        for(int i = 0; i < lists.length; i++){
            current = mergeTwoLists(current, lists[i]);
        }
        return dummy;
    }

    public int lengthOfLongestSubstring(String s) {
        int left = 0, right = 0, maxLength = 0;
        int[] charIndex = new int[128];
        Arrays.fill(charIndex, -1);

        while (right < s.length()){
            char c = s.charAt(right);
            if (charIndex[c] != -1) {
                maxLength = Math.max(maxLength, right - left);
                left = Math.max(left, charIndex[c] + 1);
            }
            charIndex[c] = right;
            right++;
        }
        if (right == s.length()){
            maxLength = Math.max(maxLength, right - left);
        }
        return maxLength;


    }

    public List<Integer> findAnagrams(String s, String p) {
        if (s.length() < p.length()){
            return new ArrayList<>();
        }
        int[] pCount = new int[26];
        int[] sCount = new int[26];
        for(int i = 0; i < p.length(); i++){
            pCount[p.charAt(i) - 'a']++;
            sCount[s.charAt(i) - 'a']++;
        }
        List<Integer> res = new ArrayList<>();
        if (Arrays.equals(pCount, sCount)){
            res.add(0);
        }
        for (int i = 0; i < s.length() - p.length(); i++){
            pCount[s.charAt(i) - 'a']--;
            pCount[s.charAt(i + p.length()) - 'a']++;
            if (Arrays.equals(pCount, sCount)){
                res.add(i + 1);
            }
        }
        return res;
    }

    public String minWindow(String s, String t) {
        if (s.length() < t.length()){
            return "";
        }

        Map<Character, Integer> tmap = new HashMap<>();
        for (char c : t.toCharArray()) {
            tmap.put(c, -1);
        }

        int min = Integer.MAX_VALUE, max = Integer.MIN_VALUE, minLength = Integer.MAX_VALUE, left = 0;
        for (int i = 0; i < s.length(); i++){
            char c = s.charAt(i);
            if (tmap.containsKey(c)){
                tmap.put(c, i);
                min = tmap.containsValue(-1) ? min : Collections.min(tmap.values());
                max = Collections.max(tmap.values());
                if (max - min >= 0 && max - min + 1 < minLength) {
                    left = min;
                    minLength = max - min + 1;
                }
            }
        }
        if (tmap.containsValue(-1)){
            return "";
        }
        return s.substring(left, left + minLength);
    }

    public boolean isPalindrome(ListNode head) {
        ListNode slow = head, fast = head;
        ListNode prev = null;

        while (fast != null && fast.next != null) {
            ListNode curr = slow;
            slow = slow.next;
            fast = fast.next.next;
            curr.next = prev;
            prev = curr;

        }

        if (fast != null) {
            slow = slow.next; // 跳过中间元素
        }

        while (prev != null && slow != null) {
            if (prev.val != slow.val) {
                return false;
            }
            prev = prev.next;
            slow = slow.next;
        }
        return true;
    }


    public int maxDepth(TreeNode root) {
        Deque<TreeNode> queue = new ArrayDeque<>();
        if (root == null) {
            return 0;
        }
        queue.offer(root);
        int depth = 0;
        while (!queue.isEmpty()){
            int size = queue.size();
            depth++;
            for (int i = 0; i < size; i++){
                TreeNode node = queue.poll();
                if (node.left != null){
                    queue.offer(node.left);
                } else if (node.right != null) {
                    queue.offer( node.right);
                }
            }
        }
        return depth;


    }


    int r = 0;
    public int diameterOfBinaryTree(TreeNode root) {
        depth(root);
        return r;

    }

    private int depth(TreeNode node){
        if (node == null){
            return 0;
        }else {
            int leftDepth = depth(node.left);
            int rightDepth = depth(node.right);
            r = Math.max(r, leftDepth + rightDepth);
            return Math.max(leftDepth, rightDepth) + 1;
        }
    }

    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        int m = nums1.length, n = nums2.length, k = m + (n - m) / 2;
        int n1 = 0, n2 = 0, left = 0, right = 0;
        for (int i = 0; i < k; i++){
            left = right;
            if (n2 >= n || (n1 < m && nums1[n1] < nums2[n2])){
                right = nums1[n1++];
            }else if(n1 >= m || (n2 < n && nums1[n1] >= nums2[n2])){
                right = nums2[n2++];
            }
        }
        if ((m + n) % 2 == 0) {
            return (left + right) / 2.0;
        } else {
            return right;
        }
    }


    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        ListNode head = new ListNode(0), cur = head;
        int sum = 0;

        while(l1 != null && l2 != null){
            sum += l1.val + l2.val;
            cur.next = new ListNode(sum % 10);
            sum = sum / 10;
            l1 = l1.next;
            l2 = l2.next;
            cur = cur.next;
        }

        if(l1 == null && l2 != null){
            cur.next = l2;
            cur = cur.next;
        }

        if(l2 == null && l1 != null){
            cur.next = l1;
            cur = cur.next;
        }

        ListNode pre = cur;

        if (l1 == null && l2 == null){
            cur = cur.next;
        }
        while(sum != 0 && cur != null){
            pre = cur;
            sum += cur.val;
            cur.val = sum % 10;
            sum = sum / 10;
            cur = cur.next;
        }

        if(sum != 0){
            pre.next = new ListNode(sum);
        }


        return head.next;

    }


    public int longestConsecutive(int[] nums) {
        int n = nums.length;
        if(n == 0 || n == 1)    return n;
        Set<Integer> numsSet = new HashSet<>();
        for(int num: nums){
            numsSet.add(num);
        }
        int maxLen = 1;

        for(int num: numsSet){
            int curLen = 1;
            if(numsSet.contains(num - 1)){
                continue;
            }
            while(numsSet.contains(num + 1)){
                curLen++;
            }
            maxLen = Math.max(maxLen, curLen);
        }

        return maxLen;

    }


    public List<List<Integer>> threeSum(int[] nums) {
        int n = nums.length;
        int i = 0;
        Arrays.sort(nums);
        List<List<Integer>> ans = new ArrayList<>();

        for(;i < n - 1; i++){
            if(nums[i] > 0){
                return ans;
            }
            if(i > 0 && nums[i] == nums[i - 1]) continue;  /// 去重
            int l = i + 1;
            int r = n - 1;

            while(l < r){
                int sum = nums[i] + nums[l] + nums[r];
                if(sum < 0){
                    l++;
                }else if(sum > 0){
                    r--;
                }else{
                    ans.add(Arrays.asList(nums[i], nums[l], nums[r]));
                    while(l < r && nums[l] == nums[l + 1]){
                        l++;
                    }
                    while(l < r && nums[r] == nums[r - 1]){
                        r--;
                    }
                     l++;
                    // r--;
                }
            }
        }
        return ans;


    }

    public int subarraySum(int[] nums, int k) {
        Arrays.sort(nums);
        int left = 0, right = left;
        int n = nums.length;
        int ans = 0;
        int sum = 0;
        if(nums[0] > k)     return 0;
        while(left < n){
            if(right >= n){
                left++;
                right = left;
                sum = 0;
                continue;
            }
            sum += nums[right];
            if(sum >= k){
                if(sum == k){
                    ans++;
                }
                left++;
                right = left;
                sum = 0;
            }else if(sum < k){
                right++;
            }

        }
        PriorityQueue <int[]> heap = new PriorityQueue<>();


        return ans;
//        Map<Integer, Integer> map = new HashMap<>();
//        map.put(0,1);
//        map.merge(0, 1, Integer::);
    }


    public int[] maxSlidingWindow(int[] nums, int k) {
        int n = nums.length;
        int[] maxVal = new int[n - k + 1];
        Deque<Integer> queue = new ArrayDeque<>();
        for(int i = 0; i < k; i++){
            while(!queue.isEmpty() && nums[i] > nums[queue.peekLast()]){
                queue.pollLast();
            }
            queue.offerLast(i);
        }
        maxVal[0] = nums[queue.peekLast()];
        for(int i = k; i < n; i++){
            while(!queue.isEmpty() && nums[i] > nums[queue.peekLast()]){
                queue.pollLast();
            }
            queue.offerLast(i);
            while(queue.peekFirst() < i - k + 1){
                queue.pollFirst();
            }

            maxVal[i - k + 1] = nums[queue.peekLast()];
        }
        return maxVal;
    }

    public List<Integer> spiralOrder(int[][] matrix) {
        int m1 = 0, n1 = -1;
        int m2 = matrix.length - 1, n2 = matrix[0].length - 1;
        List<Integer> ans = new ArrayList<>();
        int i = 0, j = 0;
        while(m1 <= m2 && n1 <= n2){
            if (!(j >= n1 && j <= n2)) break;
            while(j >= n1 && j <= n2)   ans.add(matrix[i][j++]);
            i++;
            j--;
            n1++;
            if (!(i >= m1 && i <= m2)) break;
            while(i >= m1 && i <= m2)   ans.add(matrix[i++][j]);
            j--;
            i--;
            m1++;
            if (!(j >= n1 && j <= n2)) break;
            while(j >= n1 && j <= n2)   ans.add(matrix[i][j--]);
            j++;
            i--;
            n2--;
            if (!(i >= m1 && i <= m2)) break;
            while(i >= m1 && i <= m2)   ans.add(matrix[i--][j]);
            i++;
            j++;
            m2--;
        }
        return ans;
    }

    public ListNode swapPairs(ListNode head) {
        if(head == null || head.next == null){
            return head;
        }

        ListNode newHead = head.next;
        head.next = swapPairs(newHead.next);
        newHead.next = head;
        return newHead;
    }

    public TreeNode sortedArrayToBST(int[] nums) {
        return helper(nums, 0, nums.length - 1);

    }

    public TreeNode helper(int[] nums, int left, int right){
        if(left > right){
            return null;
        }
        if(left == right){
            return new TreeNode(nums[left]);
        }
        int mid = (left + right) / 2;
        TreeNode node = new TreeNode(nums[mid]);
        node.left = helper(nums, 0, mid - 1);
        node.right = helper(nums, mid + 1, right);
        return node;
    }

    int ans;
    int count;
    public int kthSmallest(TreeNode root, int k) {
        this.count = k;
        helper(root);
        return ans;

    }

    public void helper(TreeNode root){
        if(root == null){
            return;
        }

        helper(root.left);
        count--;
        ans = root.val;
        if(count == 0){
            return;
        }

        helper(root.right);
        count--;
        ans = root.val;
        if(count == 0){
            return;
        }

    }


    TreeNode listNode = new TreeNode(0);
    TreeNode dummy = listNode;
    public void flatten(TreeNode root) {
        dfs(root);
        root = listNode.right;
    }

    public void dfs(TreeNode root){
        if(root == null){
            return;
        }
        dummy.right = new TreeNode(root.val);
//        dummy.left = null;
        dummy = dummy.right;
        dfs(root.left);
        dfs(root.right);
    }



    public int orangesRotting(int[][] grid) {
        Queue<int[]> queue = new LinkedList<>();
        int rows = grid.length, cols = grid[0].length;

        // count fresh oranges and find stale ones
        int freshCount = 0;
        for(int i = 0; i < rows; i++){
            for(int j = 0; j < cols; j++){
                if(grid[i][j] == 1){
                    freshCount++;
                }else if(grid[i][j] == 2){
                    queue.add(new int[]{i, j});
                }
            }
        }

        int minutes = 0;
        // stale oranges infect fresh ones
        while(freshCount > 0 && !queue.isEmpty()){
            minutes++;
            int r = queue.poll()[0];
            int c = queue.poll()[1];
            if(r - 1 >= 0 && grid[r - 1][c] == 1){
                freshCount--;
                grid[r - 1][c] = 2;
                queue.add(new int[]{r - 1, c});
            }

            if(r + 1 < rows && grid[r + 1][c] == 1){
                freshCount--;
                grid[r + 1][c] = 2;
                queue.add(new int[]{r + 1, c});
            }

            if(c - 1 >= 0 && grid[r][c - 1] == 1){
                freshCount--;
                grid[r][c - 1] = 2;
                queue.add(new int[]{r, c - 1});
            }

            if(c + 1 < cols && grid[r][c + 1] == 1){
                freshCount--;
                grid[r][c + 1] = 2;
                queue.add(new int[]{r, c + 1});
            }
        }

        if(freshCount <= 0){
            return minutes;
        }
        return -1;


    }

    public List<List<Integer>> permute(int[] nums) {
        int len = nums.length;
        // 使用一个动态数组保存所有可能的全排列
        List<List<Integer>> res = new ArrayList<>();
        if (len == 0) {
            return res;
        }

        boolean[] used = new boolean[len];
        List<Integer> path = new ArrayList<>();

        dfs(nums, len, 0, path, used, res);
        return res;
    }

    private void dfs(int[] nums, int len, int depth,
                     List<Integer> path, boolean[] used,
                     List<List<Integer>> res) {
        if (depth == len) {
            res.add(path);
            return;
        }

        // 在非叶子结点处，产生不同的分支，这一操作的语义是：在还未选择的数中依次选择一个元素作为下一个位置的元素，这显然得通过一个循环实现。
        for (int i = 0; i < len; i++) {
            if (!used[i]) {
                path.add(nums[i]);
                used[i] = true;

                dfs(nums, len, depth + 1, path, used, res);
                // 注意：下面这两行代码发生 「回溯」，回溯发生在从 深层结点 回到 浅层结点 的过程，代码在形式上和递归之前是对称的
                used[i] = false;
                path.remove(path.size() - 1);
            }
        }
    }


    List<List<Integer>> res = new ArrayList<>();
    List<Integer> list = new ArrayList<>();

    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        if(candidates.length == 0){
            return res;
        }
        dfs(candidates, 0, target);
        return res;


    }

    public void dfs(int[] candidates, int index, int target){
        if(index == candidates.length){
            return;
        }
        if(target == 0){
            res.add(list);
            return;
        }

        dfs(candidates, index + 1, target);
        if(target - candidates[index] >= 0){
            list.add(candidates[index]);
            dfs(candidates, index, target - candidates[index]);
            list.remove(list.size() - 1);
        }



    }

    public int largestRectangleArea(int[] heights) {
        Deque<int[]> stack = new LinkedList<>();
        int n = heights.length;
        int[] left = new int[n];
        int[] right = new int[n];
        int res = 0;

        for(int i = 0; i < n; i++ ){
            while(!stack.isEmpty() && stack.peek()[1] > heights[i]){
                stack.pop();
            }
            if(stack.isEmpty()){
                left[i] = -1;
            }
            if(!stack.isEmpty() && stack.peek()[1] < heights[i]){
                left[i] = stack.peek()[0];
            }
            stack.push(new int[]{i, heights[i]});
        }

        stack.clear();
        for(int i = n - 1; i >= 0; i--){
            while(!stack.isEmpty() && stack.peek()[1] > heights[i]){
                stack.pop();
            }
            if(stack.isEmpty()){
                left[i] = n;
            }
            if(!stack.isEmpty() && stack.peek()[1] < heights[i]){
                left[i] = stack.peek()[0];
            }
            stack.push(new int[]{i, heights[i]});
        }

        for(int i = 0; i < n; i++){
            res = Math.max(res, (right[i] - left[i] - 1) * heights[i]);
        }

        return res;

    }

    private final static Random RANDOM = new Random(System.currentTimeMillis());
    public int findKthLargest(int[] nums, int k) {
        int left = 0, right = nums.length - 1, target = nums.length - k;
        while(true){
            int index = quickSorted(nums, left, right);
            if(index == target){
                return nums[index];
            }else if(index < target){
                left = index + 1;
            }else{
                right = index - 1;
            }
        }
    }

    public int quickSorted(int[] nums, int left, int right){
        int randonIndex = left + RANDOM.nextInt(right - left + 1);
        swap(nums, left, randonIndex);
        int l = left + 1, r = right;
        int pivot = nums[left];
        while(true){
            while(l <= r && nums[l] <= pivot){
                l++;
            }
            while(l <= r && nums[r] > pivot){
                r--;
            }
            if(l > r){
                break;
            }
            swap(nums, l, r);
        }

        swap(nums, r, left);
        return r;
    }

    public void swap(int[] nums, int index1, int index2){
        int temp = nums[index1];
        nums[index1] = nums[index2];
        nums[index2] = temp;
    }

    public boolean canJump(int[] nums) {
        int maxDistance = 0;
        int n = nums.length;

        int i = 0;
        while(i < n){
            for(int j = 1; j <= nums[i]; j++){
                if(i + j >= n - 1) {
                    return true;
                }
                    maxDistance = Math.max(maxDistance, i + j + nums[i + j]);
                if(maxDistance >= n - 1){
                    return true;
                }
            }
            if(i == maxDistance){
                return false;
            }
            i = maxDistance;
        }
        return false;

    }

    public int jump(int[] nums) {
        int maxDis = 0;
        int n = nums.length;
        int start = 0, end = 1;
        int count = 0;
        int i = 0;
        while(maxDis <= n - 1){
            for(i = start; i < end; i++){
                maxDis = Math.max(maxDis, nums[i] + i);
            }
            start = end;
            end = maxDis;
            count++;
        }

        return count;
    }

    public List<Integer> partitionLabels(String s) {

        int[] last = new int[26];
        for(int i = 0; i < s.length(); i++){
            last[s.charAt(i) - 'a'] = i;
        }

        int start = 0, end = 0;
        List<Integer> res = new ArrayList<>();

        for(int i = 0; i < s.length(); i++){
            end = Math.max(end, last[s.charAt(i) - 'a']);
            if(i == end){
                res.add(end - start + 1);
            }
            start = end + 1;
        }
        return res;

    }

    public int coinChange(int[] coins, int amount) {
        int n = coins.length;
        int[][] dp = new int[n + 1][amount + 1];
        for(int i = 0; i < n + 1; i++){
            dp[i][0] = 0;
        }
        for(int i = 0; i < amount + 1; i++){
            dp[0][i] = amount + 1;
        }

        for( int i = 1; i < n + 1; i++){
            for(int j = 1; j < amount + 1; j++){
                if(j - coins[i - 1] < 0){
                    dp[i][j] = dp[i - 1][j];
                    continue;
                }
                dp[i][j] = Math.min(dp[i - 1][j], dp[i][j - coins[i-1]] + 1);
            }
        }

        return dp[n][amount];
    }

    public int minDistance(String word1, String word2) {
        int n1 = word1.length(), n2 = word2.length();

        int[][] dp = new int[n1 + 1][n2 + 1];
        for(int i = 0; i < n1; i++){
            dp[i][0] = i;
        }

        for(int j = 0; j < n2; j++ ){
            dp[0][j] = j;
        }

        for(int i = 1; i <= n1; i++){
            for(int j = 1; j <= n2; j++){
                if(word1.charAt(i - 1) == word2.charAt(j - 1)){
                    dp[i][j] = dp[i - 1][j - 1];
                    continue;
                }
                dp[i][j] = Math.min(dp[i - 1][j], dp[i][j - 1]);
                dp[i][j] = Math.min(dp[i][j], dp[i - 1][j - 1]);
                dp[i][j]++;
            }
        }
        return dp[n1][n2];

    }

    public static void main(String[] args) {
        Solution solution = new Solution();
        solution.minDistance("","a");
//        solution.coinChange(new int[]{1, 2, 5}, 11);
//        solution.partitionLabels("ababcbacadefegdehijhklij");
//        solution.jump(new int[]{1,2,3});
//        solution.canJump(new int[]{3,2,1,0,4});
//        System.out.println(solution.findKthLargest(new int[]{3,2,2,2,2}, 2));
//        solution.largestRectangleArea(new int[]{2,1,5,6,2,3});
//        solution.combinationSum(new int[]{2, 3, 5}, 8);
//        System.out.println(solution.permute(new int[]{1, 2, 3}));
//        Trie node = new Trie();
//        node.insert("apple");
//        node.search("apple");
//        System.out.println(solution.orangesRotting(new int[][]{
//                {2,1,1},
//                {1,1,0},
//                {0,1,1}
//        }));
//        solution.flatten(new TreeNode(1, new TreeNode(2, new TreeNode(3), new TreeNode(4)), new TreeNode(5, null , new TreeNode(6))));
//        solution.sortedArrayToBST(new int[]{-10, -3, 0, 5, 9});
//        solution.swapPairs(new ListNode(1, new ListNode(2, new ListNode(3, new ListNode(4)))));
//        System.out.println(solution.spiralOrder(new int[][]{
//                {1, 2, 3,4},
//                {5, 6, 7,8},
//                {9, 10, 11,12}
//
//        }));
//        System.out.println(Arrays.toString(solution.maxSlidingWindow(new int[]{1,3,-1,-3,5,3,6,7}, 3)));
//        System.out.println(solution.subarraySum(new int[]{1,1,1}, 2));
//        System.out.println(solution.longestConsecutive(new int[]{0, 3, 7, 2, 5, 8, 4, 6, 0, 1}));
//        System.out.println(solution.isPalindrome(new ListNode(1, new ListNode(2, new ListNode(2, new ListNode(1))))));
//        System.out.println(solution.diameterOfBinaryTree(new TreeNode(3, new TreeNode(9), new TreeNode(20, new TreeNode(15), new TreeNode(7)))));
//        System.out.println(solution.findMedianSortedArrays(new int[]{1, 3}, new int[]{2}));
        /*System.out.println(solution.addTwoNumbers(
                new ListNode(9, new ListNode(9, new ListNode(9, new ListNode(9, new ListNode(9, new ListNode(9, new ListNode(9))))))),
                new ListNode(9, new ListNode(9, new ListNode(9, new ListNode(9))))
        ));*/
//        System.out.println(solution.addTwoNumbers(
//                new ListNode(5),
//                new ListNode(5)
//        ));
        Deque<Integer> deque = new LinkedList<Integer>();
        Deque<Integer> deque1 = new ArrayDeque<>();
        List<int[]> list = new ArrayList<>();
        list.isEmpty();
        list.remove(0);

        Map<Integer, Integer> map = new LinkedHashMap<>();
        Map<Integer, Integer> map1 = new HashMap<>();
        Map<Integer, Integer> treeMap = new TreeMap<>();
        Set<Integer> set = new TreeSet<>();
        Iterator iterator = set.iterator();
        String s = new String();
        s.toCharArray();
        StringBuilder sb = new StringBuilder("");
        sb.deleteCharAt(sb.length() - 1);
        sb.append("");
        sb.replace(0,1,"1");
        String string = "12";
        int nu = Integer.parseInt(string);
        String str = "123";
        int i = Integer.valueOf('c');
        Deque<int[]> stack = new LinkedList<>();
        stack.push(new int[]{1, 2});
        stack.clear();
        PriorityQueue<Integer> heap = new PriorityQueue<>(1, (a,b)->b - a);
        heap.offer(2);
        heap.toArray();
        heap.size();
        heap.poll();
        heap.peek();
        map1.keySet();

        Scanner in = new Scanner(System.in);
        in.hasNextInt();

        sb.append(1);
        sb.append('1');
        String.valueOf(s.charAt(0)).toUpperCase();
        sb.insert(0,"1");

        int n = Integer.valueOf("10");
        int[] arr = new int[4];
        arr = new int[]{n, n, n, n};
        List<Integer> list1 = new ArrayList<>();




    }



}
//leetcode submit region end(Prohibit modification and deletion)

class Trie {
    private Trie[] children;
    private boolean isEnd;

    public Trie() {
        children = new Trie[26];
        isEnd = false;

    }

    public void insert(String word) {
        Trie node = this;
        for(int i = 0; i < word.length(); i++){
            int index = word.charAt(i) - 'a';
            if(node.children[index] == null){
                node.children[index] = new Trie();
            }
            node = node.children[index];
        }
        node.isEnd = true;

    }

    public boolean search(String word) {
        Trie node = this;
        for(int i = 0; i < word.length(); i++){
            int index = word.charAt(i) - 'a';
            if(node.children[index] == null){
                return false;
            }
            node = node.children[index];
        }
        return node.isEnd;
    }

    public boolean startsWith(String prefix) {
        Trie node = this;
        for(int i = 0; i < prefix.length(); i++){
            int index = prefix.charAt(i) - 'a';
            if(node.children[index] == null){
                return false;
            }
            node = node.children[index];
        }
        return true;

    }


}