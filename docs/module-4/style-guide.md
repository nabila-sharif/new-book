# Vision-Language-Action (VLA) Module Documentation Style Guide

## Purpose

This document defines the consistent documentation format and style guidelines for Module 4: Vision-Language-Action (VLA) to ensure consistency with earlier modules in the Physical AI & Humanoid Robotics book.

## Document Structure

### Standard Frontmatter

Each document should include the following frontmatter:

```markdown
---
sidebar_position: X
title: 'Chapter X: [Chapter Title]'
---
```

Where X is the appropriate sidebar position number.

### Standard Document Sections

Each chapter should follow this structure:

1. **Learning Objectives** - Clear, measurable objectives using "By the end of this chapter, you will be able to:" format
2. **Key Topics** - Main content sections with hierarchical headings
3. **Practical Implementation** - Code examples, tutorials, and hands-on content
4. **Assessment Criteria** - Measurable outcomes for the chapter

## Writing Style

### Tone and Voice

- Use an educational, professional tone appropriate for technical documentation
- Write in active voice wherever possible
- Address the reader as "you" to create an engaging learning experience
- Use clear, concise language avoiding unnecessary jargon

### Technical Terminology

- Define technical terms when first introduced
- Use consistent terminology throughout all VLA module documents
- When introducing new terms, use italics for the first occurrence: *term*
- Acronyms should be spelled out on first use: "Vision-Language-Action (VLA)"

### Code Examples

- Use Python for all code examples as specified in the module requirements
- Include appropriate language tags for syntax highlighting: ```python
- Use realistic but simple examples that demonstrate key concepts
- Include comments in code to explain complex operations
- Follow PEP 8 style guidelines for Python code

## Formatting Standards

### Headings

- Use # for main title (document title)
- Use ## for major sections (Learning Objectives, Key Topics, etc.)
- Use ### for subsections
- Use #### sparingly for detailed subcategories

### Lists

- Use hyphens (-) for unordered lists
- Use numbers (1., 2., 3.) for ordered lists
- Maintain consistent indentation (2 spaces)

### Code and Technical Elements

- Use single backticks (`) for inline code and technical terms
- Use triple backticks (```) for code blocks with appropriate language identifier
- Use **bold** for emphasis when needed
- Use *italics* for new terms or emphasis

## Content Guidelines

### Learning Objectives

- Each objective should be specific, measurable, and actionable
- Use action verbs: "implement", "design", "create", "integrate", "apply"
- Limit to 4-6 objectives per chapter
- Format as: "By the end of this chapter, you will be able to:"

### Practical Implementation

- Provide complete, runnable code examples
- Include explanations of how code relates to VLA concepts
- Use real-world robotics scenarios when possible
- Include troubleshooting tips and common pitfalls

### Assessment Criteria

- Create measurable, specific criteria
- Align with learning objectives
- Use action-oriented language
- Format as: "- Students can [action] [object] [context]"

## Cross-Chapter Consistency

### Terminology Consistency

- VLA = Vision-Language-Action
- LLM = Large Language Model
- ROS 2 = Robot Operating System 2
- Whisper = OpenAI Whisper (first use), Whisper (subsequent uses)

### Code Pattern Consistency

- Use consistent variable naming conventions
- Follow similar architectural patterns across chapters
- Maintain consistent ROS topic and service naming

## Visual Elements

- Use diagrams where they add value (not just decoration)
- Include code output examples when relevant
- Use tables for comparing options or showing data
- Ensure all visual elements have descriptive alt text

## Linking and Navigation

- Link to related sections within the VLA module when relevant
- Use relative links for internal navigation
- Include "See also" sections for related topics
- Maintain clear breadcrumbs to previous and next sections
