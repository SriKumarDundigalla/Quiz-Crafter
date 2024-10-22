def process_questions(questions):
    structured_questions = []
    for q in questions:
        parts = q.split("#")
        question_text = parts[0].strip("*")
        options = parts[1:-1]
        answer = parts[-1].split("**Answer: ")[1].strip("**")

        question_data = {
            "question": question_text,
            "options": options,
            "answer": answer
        }
        structured_questions.append(question_data)
    
    return structured_questions

questions = [
    "**Question1. What determines the color of the points in the scatter plot?**#A) The size of the point#B) The x-coordinate of the point#C) The y-coordinate of the point#D) The label of the plot#**Answer: C) The y-coordinate of the point**",
    "**Question2. How is the size of each point in the scatter plot determined?**#A) Based on its x-coordinate#B) Based on its y-coordinate#C) Randomly assigned#D) All points have the same size#**Answer: C) Randomly assigned**",
    "**Question3. What does the line 'plt.axhline(y=50, color='r', linestyle='--')' represent in the plot?**#A) A vertical line at x=50#B) A horizontal line at y=50#C) A trend line for the scatter plot#D) A boundary separating different colors of points#**Answer: B) A horizontal line at y=50**",
    "**Question4. Which of the following best describes the purpose of 'plt.plot([0, 100], [0, 200], linestyle='--', lw=2)' in the code?**#A) To draw a scatter plot#B) To draw a horizontal line at y=200#C) To draw a line with a slope of 2 starting from the origin#D) To outline the plot area#**Answer: C) To draw a line with a slope of 2 starting from the origin**",
    "**Question5. What is the effect of the 'marker=\"o\"' parameter in the plt.scatter function?**#A) It changes the plot type to a line plot#B) It specifies the shape of the markers to be circles#C) It increases the size of the markers#D) It colors the markers based on their value#**Answer: B) It specifies the shape of the markers to be circles**"
]

processed_questions = process_questions(questions)
print(processed_questions)
