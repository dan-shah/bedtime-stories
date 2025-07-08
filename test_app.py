import streamlit as st

st.title("Simple Test App")
st.write("If you can see this, Streamlit is working!")

if st.button("Click me"):
    st.write("Button clicked!")