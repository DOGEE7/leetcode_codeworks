package leetcode.editor;

import leetcode.editor.util.*;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

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

    public static void main(String[] args) {
        Solution solution = new Solution();
        System.out.println(solution.findAnagrams("baa","aa"));
    }
}
//leetcode submit region end(Prohibit modification and deletion)

